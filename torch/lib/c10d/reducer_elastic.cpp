#include <c10d/reducer.hpp>
#include <c10d/reducer_elastic.hpp>

#include <functional>

#include <c10/core/DeviceGuard.h>
#include <c10/core/StreamGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/hash.h>
#include <c10d/comm.hpp>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/utils/grad_layout_contract.h>
#include <torch/csrc/autograd/utils/lambda_post_hook.h>
#include <torch/csrc/utils/memory.h>

#include <ctime>

namespace c10d {
namespace {

inline int64_t current_time_in_nanos() {
  return torch::autograd::profiler::getTime();
}

constexpr int kUnsetDivFactor = -1;

} // namespace

ElasticReducer::ElasticReducer(
    std::vector<std::vector<torch::autograd::Variable>> replicas,
    std::vector<std::vector<size_t>> bucket_indices,
    c10::intrusive_ptr<c10d::ProcessGroup> process_group,
    c10::intrusive_ptr<c10d::ProcessGroup> process_group_comp,
    std::vector<std::vector<bool>> expect_sparse_gradients,
    int64_t bucket_bytes_cap,
    bool find_unused_parameters,
    bool gradient_as_bucket_view,
    bool is_computation_process,
    int64_t logic_gpus_num,
    int64_t local_rank,
    int64_t cheif_rank,
    at::Tensor shm_array,
    std::vector<at::Tensor> shm_grads)
    : replicas_(std::move(replicas)),
      process_group_(std::move(process_group)),
      process_group_comp_(std::move(process_group_comp)),
      expect_sparse_gradients_(std::move(expect_sparse_gradients)),
      expect_autograd_hooks_(false),
      require_finalize_(false),
      next_bucket_(0),
      has_marked_unused_parameters_(false),
      find_unused_parameters_(find_unused_parameters),
      gradient_as_bucket_view_(gradient_as_bucket_view),
      local_used_maps_reduced_(false),
      backward_stats_base_(0),
      has_rebuilt_bucket_(false),
      bucket_bytes_cap_(bucket_bytes_cap),
      divFactor_(kUnsetDivFactor),
      is_computation_process_(is_computation_process),
      local_rank_(local_rank),
      cheif_rank_(cheif_rank),
      cntMicroBatch_(-1),
      logic_gpus_num_(logic_gpus_num),
      shm_array_(shm_array),
      shm_grads_(shm_grads),
      comm_hook_(nullptr),
      ddp_logging_data_(std::move(std::make_unique<c10::DDPLoggingData>())) {
  C10_LOG_API_USAGE_ONCE("torch.distributed.ddp.reducer_elastic");
  TORCH_CHECK(replicas_.size() == 1, "Expected only one model replica in ElasticReducer.");
  TORCH_CHECK(replicas_[0].size() >= 1, "Expected at least one parameter.");

  // If `expect_sparse_gradients` is not specified, initialize it such that
  // we do not expect sparse gradients for any parameter.
  if (expect_sparse_gradients_.empty()) {
    expect_sparse_gradients_ = std::vector<std::vector<bool>>(
        replicas_.size(), std::vector<bool>(replicas_[0].size(), false));
  }
  TORCH_INTERNAL_ASSERT(expect_sparse_gradients_.size() == replicas_.size());

  // Corresponding params' layouts (strides) must match across
  // replicas within this process and across processes.
  // (see Note:  "Gradient Layout Contract" in initialize_buckets).
  verify_replicas_within_process();
  if (is_computation_process_)
    verify_replica0_across_processes();

  // Initialize variable bucketing.
  // This can be reinitialized later after capturing runtime information.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    initialize_buckets(std::move(bucket_indices));
  }

  // All variables are expected to have their `grad_fn` set to the gradient
  // accumulation function (since they are leafs in the autograd graph).
  // We store pointers to these functions such that we can check if they are
  // used in an autograd pass. If they are not, we know their grad tensors
  // can be marked as ready for reduction.
  if (is_computation_process_)
  {
    const auto replica_count = replicas_.size();
    grad_accumulators_.resize(replica_count);
    for (size_t replica_index = 0; replica_index < replica_count;
         replica_index++) {
      const auto variable_count = replicas_[replica_index].size();
      grad_accumulators_[replica_index].resize(variable_count);
      for (size_t variable_index = 0; variable_index < variable_count;
           variable_index++) {
        auto& variable = replicas_[replica_index][variable_index];
        const auto index = VariableIndex(replica_index, variable_index);

        // The gradient accumulator function is lazily initialized once.
        // Therefore we can use its presence in the autograd graph as
        // evidence that the parameter has participated in an iteration.
        auto grad_accumulator =
            torch::autograd::impl::grad_accumulator(variable);

#ifndef _WIN32
        using torch::distributed::autograd::ThreadLocalDistAutogradContext;
#endif
        // Hook to execute after the gradient accumulator has executed.
        hooks_.emplace_back(
            grad_accumulator->add_post_hook(
                torch::make_unique<torch::autograd::utils::LambdaPostHook>(
                    [=](const torch::autograd::variable_list& outputs,
                        const torch::autograd::variable_list& /* unused */) {
#ifndef _WIN32
                      this->rpc_context_.set(
                          ThreadLocalDistAutogradContext::getContextPtr());
#endif
                      // this->autograd_hook(index);
                      this->autograd_hook_compProcess(index);
                      return outputs;
                    })),
            grad_accumulator);

        // Map raw function pointer to replica index and parameter index.
        // This is used later on when the autograd graph is traversed
        // to check for parameters for which no gradient is computed, if
        // find_unused_parameters=True.
        // We maintain a mapping of gradient accumulator to vector of variables,
        // since multiple parameters may share the same grad accumulator.
        if (find_unused_parameters_) {
          auto gradAcc = gradAccToVariablesMap_.find(grad_accumulator.get());
          if (gradAcc == gradAccToVariablesMap_.end()) {
            std::vector<VariableIndex> indexVec{index};
            gradAccToVariablesMap_[grad_accumulator.get()] =
                std::move(indexVec);
          } else {
            // Scenario where we have indices whose corresponding parameters
            // share the same grad accumulator.
            gradAcc->second.push_back(index);
          }
        }

        // The gradient accumulator is stored as weak_ptr in the autograd
        // metadata of the variable, so we have to keep it alive here for
        // the raw pointer to be valid.
        grad_accumulators_[replica_index][variable_index] =
            std::move(grad_accumulator);
      }
    }
  }

  // Initialize backward stats vector.
  // Comment(Mingzhen) Skipped by the communication process
  if (is_computation_process_)
  {
    const auto replica_count = replicas_.size();
    backward_stats_.resize(replica_count);
    const auto variable_count = replicas_[0].size();
    std::for_each(
        backward_stats_.begin(),
        backward_stats_.end(),
        [=](std::vector<int64_t>& v) { v.resize(variable_count); });
  }

  // See Note [Skip allreducing local_used_maps_dev]
  // Comment(Mingzhen) Skipped by the communication process
  if (is_computation_process_)
  if (find_unused_parameters_) {
    // Initialize locally used parameter maps
    {
      const auto replica_count = replicas_.size();
      const auto variable_count = replicas_[0].size();
      local_used_maps_.resize(replica_count);
      local_used_maps_dev_.resize(replica_count);

      for (size_t i = 0; i < replica_count; i++) {
        at::TensorOptions options;
        options = options.dtype(at::kInt);

        if (replicas_[i][0].is_cuda()) {
          at::DeviceGuard g(replicas_[i][0].device());
          local_used_maps_[i] = at::zeros(
              {static_cast<long>(variable_count)}, options.pinned_memory(true));
        } else {
          local_used_maps_[i] =
              at::zeros({static_cast<long>(variable_count)}, options);
        }

        // This tensor needs to be on the same device as replica because backend
        // such as NCCL may not support CPU tensors, and hence it might not work
        // if we always put it on CPU.
        options = options.device(replicas_[i][0].device());
        local_used_maps_dev_[i] =
            at::empty({static_cast<long>(variable_count)}, options);
      }
    }
  }
}

// Note [Skip allreducing local_used_maps_dev]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// If find_unused_parameters_ is set to false, there is no need to allreduce
// local_used_maps_dev_, because all parameters will be reduced anyway.
// Therefore, we can avoid allocating memory for local_used_maps and
// local_used_maps_dev_ if find_unused_parameters_ is false.

// Note [DDP Communication Hook]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// If DDP communication hook is not registered, the reducer reduces the buckets
// by just calling allreduce. If registered, it calls the hook and uses future
// work handle. If registered, reducer also skips dividing grads by world size.
// The reason for this is that the communication hook is expected to completely
// override how we perform communication and the user should have complete
// control over how the grads are handled.
//
// DDP communication hook is an enhancement that provides a hook which can be
// used to override how DDP communicates gradients across ranks, this can be
// used for algorithms like Gradient Compression/GossipGrad. This hook can be
// registered from Python API using `register_comm_hook`. `PythonCommHook`
// enables registering a Python hook and is a subclass of `CommHookInterface`.
// Additionally, there are also some built-in C++ hook implementations that can
// be specified by calling `register_builtin_comm_hook` from Python API.

ElasticReducer::~ElasticReducer() noexcept(false) {
  // Remove all hooks on variables registered by this ElasticReducer. This is necessary
  // to make DDP failure recoverable. Otherwise, multiple ElasticReducer instances
  // (from recoveries) will add their hooks to the original model, and those
  // hooks will try to invoke methods on a deleted ElasticReducer objects.
  for (auto& hook : hooks_) {
    auto& key = hook.first;
    auto& grad_accumulator = hook.second;
    TORCH_CHECK(
        grad_accumulator->del_post_hook(key),
        "ElasticReducer attempts to delete a non-existing hook.");
  }
}

// Verifies replicas in this process treat the same number of params,
// all params require grad, and corresponding params across replicas
// have the same dtype/size/layout.
void ElasticReducer::verify_replicas_within_process() {
  const auto replica_count = replicas_.size();
  for (size_t replica_index = 0; replica_index < replica_count;
       replica_index++) {
    const auto variable_count = replicas_[replica_index].size();
    TORCH_CHECK(
        replicas_[replica_index].size() == replicas_[0].size(),
        "Model replicas must have an equal number of parameters.");
    TORCH_CHECK(
        expect_sparse_gradients_[replica_index].size() ==
            expect_sparse_gradients_[0].size(),
        "Expected number of entries in expect_sparse_gradients ",
        "to be equal across replicas.");
    for (size_t variable_index = 0; variable_index < variable_count;
         variable_index++) {
      TORCH_CHECK(
          replicas_[replica_index][variable_index].requires_grad(),
          "Variables must require gradients (have `requires_grad` set).");
      TORCH_CHECK(
          replicas_[replica_index][variable_index].sizes() ==
              replicas_[0][variable_index].sizes(),
          "Variables across model replicas must have identical sizes.");
      TORCH_CHECK(
          replicas_[replica_index][variable_index].strides() ==
              replicas_[0][variable_index].strides(),
          "Variables across model replicas must have identical strides.");
      TORCH_CHECK(
          replicas_[replica_index][variable_index].dtype() ==
              replicas_[0][variable_index].dtype(),
          "Variables across model replicas must have identical dtype.");
      TORCH_CHECK(
          expect_sparse_gradients_[replica_index][variable_index] ==
              expect_sparse_gradients_[0][variable_index],
          "Expected the same variables across replicas to either both ",
          "or neither expect a sparse gradient.");
    }
  }
}

// Verifies corresponding params in replica 0 have the same sizes/strides
// across processes.
void ElasticReducer::verify_replica0_across_processes() {
  size_t i = 0;
  for (const auto& t : replicas_[0]) {
    i += 2 * t.dim();
  }
  at::TensorOptions options;
  options = options.dtype(at::kLong);
  auto metadata = at::empty({static_cast<long>(i)}, options);

  // Technically, process 0 is the broadcast source, so only process 0 needs
  // to populate metadata.  But no harm keeping work aligned across processes.
  auto metadata_accessor = metadata.accessor<int64_t, 1>();
  i = 0;
  for (const auto& t : replicas_[0]) {
    for (const auto& sz : t.sizes()) {
      metadata_accessor[i++] = sz;
    }
    for (const auto& str : t.strides()) {
      metadata_accessor[i++] = str;
    }
  }

  auto metadata_dev = metadata.clone().to(replicas_[0][0].device());
  std::vector<at::Tensor> vec{metadata_dev};
  process_group_comp_->broadcast(vec)->wait();

  // Technically, process 0 doesn't need to double-check metadata, because it
  // was the source.  But no harm keeping work aligned.
  auto control = at::empty({static_cast<long>(i)}, options);
  control.copy_(metadata_dev, /*non_blocking=*/false);
  auto control_accessor = control.accessor<int64_t, 1>();
  i = 0;
  for (size_t p = 0; p < replicas_[0].size(); p++) {
    const auto& t = replicas_[0][p];
    // I'd like to include which process we are in the message,
    // but ProcessGroup::getRank is not public!
    for (const auto& sz : t.sizes()) {
      TORCH_CHECK(
          sz == control_accessor[i++],
          "replicas[0][",
          p,
          "] in this process"
          " with sizes ",
          t.sizes(),
          " appears not to match sizes of the same param in process 0.");
    }
    for (const auto& str : t.strides()) {
      TORCH_CHECK(
          str == control_accessor[i++],
          "replicas[0][",
          p,
          "] in this process"
          " with strides ",
          t.strides(),
          " appears not to match strides of the same param in process 0.");
    }
  }
}

void ElasticReducer::check_grad_layout(
    const at::Tensor& grad,
    const at::Tensor& bucket_view) {
  // Ensure that the gradient type matches the bucket type.
  TORCH_CHECK(
      grad.options().type_equal(bucket_view.options()),
      "Expected ",
      bucket_view.toString(),
      ", got ",
      grad.toString());
  TORCH_INTERNAL_ASSERT(grad.device() == bucket_view.device());
  TORCH_INTERNAL_ASSERT(grad.numel() == bucket_view.numel());
  // AccumulateGrad doesn't HAVE to obey the grad layout contract.
  // The penalty for disobedience is reduced performance, not numerical
  // death. Warnings here help diagnose poor DDP performance.
  if (grad.strides() != bucket_view.strides()) {
    TORCH_WARN_ONCE(
        "Grad strides do not match bucket view strides. "
        "This may indicate grad was not created according to the "
        "gradient layout contract, or that the param's strides "
        "changed since DDP was constructed.  This is not an error, "
        "but may impair performance.\n"
        "grad.sizes() = ",
        grad.sizes(),
        ", strides() = ",
        grad.strides(),
        "\n",
        "bucket_view.sizes() = ",
        bucket_view.sizes(),
        ", strides() = ",
        bucket_view.strides());
  }
  if (!gradient_as_bucket_view_) {
    TORCH_INTERNAL_ASSERT(!grad.is_alias_of(bucket_view));
  }
}

void ElasticReducer::copy_grad_to_bucket(
    const at::Tensor& grad,
    at::Tensor& bucket_view) {
  // See Note [DDP Communication Hook]
  if (comm_hook_ == nullptr) {
    // imitates wrapped_scalar_tensor in ATen/native/BinaryOps.cpp
    auto wrapped = c10::scalar_to_tensor(double(1.) / divFactor_);
    wrapped.unsafeGetTensorImpl()->set_wrapped_number(true);
    // Divides while copying into the bucket view.
    at::native::mul_out(bucket_view, grad, wrapped);
  } else {
    bucket_view.copy_(grad);
  }
}

void ElasticReducer::mark_variable_ready_dense(VariableIndex index) {
  const auto replica_index = index.replica_index;
  const auto variable_index = index.variable_index;
  const auto& bucket_index = variable_locators_[variable_index];
  auto& bucket = buckets_[bucket_index.bucket_index];
  auto& replica = bucket.replicas[replica_index];
  auto& variable = replica.variables[bucket_index.intra_bucket_index];
  const auto offset = replica.offsets[bucket_index.intra_bucket_index];
  const auto length = replica.lengths[bucket_index.intra_bucket_index];
  auto& bucket_view = replica.bucket_views_in[bucket_index.intra_bucket_index];

  // Copy contents of gradient tensor to bucket tensor.
  // If the gradient is not set, we assume it wasn't computed
  // as part of the current backwards pass, and zero the part
  // of the bucket it would otherwise hold.
  runGradCallbackForVariable(variable, [&](auto& grad) {
    if (grad.defined()) {
      this->check_grad_layout(grad, bucket_view);
      // When gradient_as_bucket_view_ is false, or even when
      // gradient_as_bucket_view_ is true, in rare cases users may set grad to
      // be None after every iteration. In these cases, grad and bucket_view are
      // pointing to different storages and thus need to copy grads to
      // bucket_view. If gradient_as_bucket_view_ is set as true, let grad point
      // to bucket_view. If grad has already been set as views of buckets in
      // previous iterations, no copy is needed.
      if (!grad.is_alias_of(bucket_view)) {
        this->copy_grad_to_bucket(grad, bucket_view);
        if (gradient_as_bucket_view_) {
          // Let grad point to bucket_view buffer.
          grad = bucket_view;
          // The grad is modified and need to be written back.
          return true;
        }
      } else {
        // If grad and bucket view point to the same storage, no need to copy
        if (comm_hook_ == nullptr) {
          bucket_view.div_(divFactor_);
        }
      }
    } else {
      bucket_view.zero_();
    }
    // The grad is not modified and doesn't need to be written back.
    return false;
  });
}

void ElasticReducer::mark_variable_ready_sparse(VariableIndex index) {
  const auto replica_index = index.replica_index;
  const auto variable_index = index.variable_index;
  const auto& bucket_index = variable_locators_[variable_index];
  auto& bucket = buckets_[bucket_index.bucket_index];
  auto& replica = bucket.replicas[replica_index];
  auto& variable = replica.variables[bucket_index.intra_bucket_index];

  runGradCallbackForVariable(variable, [&](auto& grad) {
    TORCH_CHECK(grad.defined(), "Expected sparse gradient to be defined.");
    TORCH_CHECK(
        grad.options().layout() == c10::kSparse,
        "Expected variable to have sparse gradient.");

    // Sparse tensors cannot be grouped together with other sparse tensors
    // in a single reduction operation like we can for dense tensors.
    // Therefore, the `offsets` and `lengths` vectors in the bucket replica
    // struct are empty, and there is no pre-existing accumulation tensor.
    // Directly assign the sparse tensor to the `contents` field.
    replica.contents = grad;
    // See Note [DDP Communication Hook]
    if (comm_hook_ == nullptr) {
      replica.contents.div_(divFactor_);
    }
    // The grad is modified in place and needs to be written back.
    return true;
  });
}

std::vector<std::vector<at::Tensor>> ElasticReducer::get_bucket_tensors() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<std::vector<at::Tensor>> bucketTensors;
  bucketTensors.reserve(buckets_.size());
  for (const auto& bucket : buckets_) {
    std::vector<at::Tensor> tensors;
    tensors.reserve(bucket.replicas.size());
    for (const auto& rep : bucket.replicas) {
      tensors.push_back(rep.contents);
    }
    bucketTensors.push_back(std::move(tensors));
  }
  return bucketTensors;
}

void ElasticReducer::set_forward_pass_work_handle(
    c10::intrusive_ptr<c10d::ProcessGroup::Work> forwardPassWorkHandle,
    bool useStaticWorldSize) {
  std::lock_guard<std::mutex> lock(mutex_);
  forwardPassWorkHandle_.workHandle = std::move(forwardPassWorkHandle);
  forwardPassWorkHandle_.useStaticWorldSize = useStaticWorldSize;
}

std::vector<at::Tensor> ElasticReducer::get_local_used_maps_on_device() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return local_used_maps_dev_;
}

void ElasticReducer::push_rebuilt_params_for_all_indices() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!should_rebuild_buckets() || !rebuilt_param_indices_.empty()) {
    return;
  }
  const auto replica_count = replicas_.size();
  for (size_t replica_index = 0; replica_index < replica_count;
       ++replica_index) {
    const auto variable_count = replicas_[replica_index].size();
    for (size_t variable_index = 0; variable_index < variable_count;
         ++variable_index) {
      const auto index = VariableIndex(replica_index, variable_index);
      push_rebuilt_params(index);
    }
  }
}

void ElasticReducer::push_rebuilt_params(const VariableIndex& index) {
  if (should_rebuild_buckets() && index.replica_index == 0) {
    rebuilt_params_.push_back(
        replicas_[index.replica_index][index.variable_index]);
    rebuilt_param_indices_.push_back(index.variable_index);
  }
}

// Comment(Mingzhen) A wrapper of the autograd_hook()
void ElasticReducer::autograd_hook_compProcess(VariableIndex index) {
  if (cntMicroBatch_ == (logic_gpus_num_-1) ) {
    autograd_hook(index);
  }
}

// The function `autograd_hook` is called after the gradient for a
// model parameter has been accumulated into its gradient tensor.
// This function is only to be called from the autograd thread.
void ElasticReducer::autograd_hook(VariableIndex index) {
  std::lock_guard<std::mutex> lock(this->mutex_);

  // See Note [Skip allreducing local_used_maps_dev]
  if (find_unused_parameters_) {
    // Since it gets here, this param has been used for this iteration. We want
    // to mark it in local_used_maps_. During no_sync session, the same var can
    // be set multiple times, which is OK as does not affect correctness. As
    // long as it is used once during no_sync session, it is marked as used.
    local_used_maps_[index.replica_index][index.variable_index] = 1;
  }

  // Ignore if we don't expect to be called.
  // This may be the case if the user wants to accumulate gradients
  // for number of iterations before reducing them.
  if (!expect_autograd_hooks_) {
    return;
  }

  // Rebuild bucket only if 1) it is the first time to rebuild bucket 2)
  // find_unused_parameters_ is false, currently it does not support when there
  // are unused parameters 3) this backward pass needs to run allreduce. Here,
  // we just dump tensors and their parameter indices into rebuilt_params_ and
  // rebuilt_param_indices_ based on gradient arriving order, and then at the
  // end of finalize_backward(), buckets will be rebuilt based on
  // rebuilt_params_ and rebuilt_param_indices_, and then will be broadcasted
  // and initialized. Also we only need to dump tensors and parameter indices of
  // one replica.
  push_rebuilt_params(index);

  // If `find_unused_parameters_` is true there may be model parameters that
  // went unused when computing the model output, they won't be part of the
  // autograd graph, and won't receive gradients. These parameters are
  // discovered in the `prepare_for_backward` function and their indexes stored
  // in the `unused_parameters_` vector.
  if (!has_marked_unused_parameters_ && find_unused_parameters_) {
    has_marked_unused_parameters_ = true;
    for (const auto& unused_index : unused_parameters_) {
      mark_variable_ready(unused_index);
    }
  }

  // Finally mark variable for which this function was originally called.
  mark_variable_ready(index);
}

void ElasticReducer::mark_variable_ready(VariableIndex index) {
  const auto replica_index = index.replica_index;
  const auto variable_index = index.variable_index;
  TORCH_CHECK(replica_index < replicas_.size(), "Out of range replica index.");
  TORCH_CHECK(
      variable_index < variable_locators_.size(),
      "Out of range variable index.");
  backward_stats_[replica_index][variable_index] =
      current_time_in_nanos() - backward_stats_base_;

  // Any time we mark a variable ready (be it in line due to unused parameters,
  // or via an autograd hook), we require a call to the finalize function. If
  // this doesn't happen before the next iteration (or call to
  // `prepare_for_backwards`), we know something is wrong.
  require_finalize_ = true;

  const auto& bucket_index = variable_locators_[variable_index];
  auto& bucket = buckets_[bucket_index.bucket_index];
  auto& replica = bucket.replicas[replica_index];

  // Something is wrong if all variables contained in this bucket replica have
  // already been marked as ready.
  if (replica.pending == 0) {
    const auto common_error = c10::str(
        "Expected to mark a variable ready only once. ",
        "",
        "This error is caused by one of the following reasons: ",
        "1) Use of a module parameter outside the `forward` function. ",
        "Please make sure model parameters are not shared across multiple ",
        "concurrent forward-backward passes",
        "2) Reused parameters in multiple reentrant backward passes. For ",
        "example, if you use multiple `checkpoint` functions to wrap the ",
        "same part of your model, it would result in the same set of ",
        "parameters been used by different reentrant backward passes ",
        "multiple times, and hence marking a variable ready multiple times. ",
        "DDP does not support such use cases yet.");
    TORCH_CHECK(
        has_marked_unused_parameters_,
        common_error,
        "3) Incorrect unused parameter detection. The return value of the ",
        "`forward` function is inspected by the distributed data parallel ",
        "wrapper to figure out if any of the module's parameters went ",
        "unused. For unused parameters, DDP would not expect gradients from ",
        "then. However, if an unused parameter becomes part of the autograd ",
        "graph at a later point in time (e.g., in a reentrant backward when ",
        "using `checkpoint`), the gradient will show up unexpectedly. If all ",
        "parameters in the model participate in the backward pass, you can ",
        "disable unused parameter detection by passing the keyword argument ",
        "`find_unused_parameters=False` to ",
        "`torch.nn.parallel.DistributedDataParallel`.");
    TORCH_CHECK(!has_marked_unused_parameters_, common_error);
  }

  // If it was scheduled, wait on allreduce in forward pass that tells us
  // division factor based on no. of currently participating processes.
  if (divFactor_ == kUnsetDivFactor) {
    divFactor_ = process_group_->getSize();
    auto& workHandle = forwardPassWorkHandle_.workHandle;
    if (workHandle && !forwardPassWorkHandle_.useStaticWorldSize) {
      workHandle->wait();
      auto results = workHandle->result();
      // Guard against the results being empty
      TORCH_INTERNAL_ASSERT(results.size() > 0);
      at::Tensor& res = results.front();
      divFactor_ = res.item().to<int>();
    }
  }

  if (bucket.expect_sparse_gradient) {
    mark_variable_ready_sparse(index);
  } else {
    mark_variable_ready_dense(index);
  }

  // TODO(@pietern): Make this work for both CPU/CUDA tensors.
  // When using CPU tensors we don't need to do this.
  // // Record event so that we can wait for all of them.
  // auto& event = replica.events[bucket_index.intra_bucket_index];
  // event.record();

  // Check if this was the final gradient for this bucket.
  if (--replica.pending == 0) {
    // Kick off reduction if all replicas for this bucket are ready.
    if (--bucket.pending == 0) {
      mark_bucket_ready(bucket_index.bucket_index);
    }
  }

  // Run finalizer function and kick off reduction for local_used_maps once the
  // final bucket was marked ready.
  if (next_bucket_ == buckets_.size()) {

    if (is_computation_process_) {
      if ( (!is_bucketOrder_dumped_) && (process_group_->getSize() > 1) ) {

        auto shm_array = shm_array_.accessor<int, 2>();
        for (int proc = 0; proc < logic_gpus_num_-1; proc++) {
          shm_array[proc][1] = ORDER_DUMPED;
        }
        customized_broadcast(bucket_allreduce_order_, cheif_rank_);
        cnt_bucketOrder_ += 1;
        if (cnt_bucketOrder_ >= 2) {
          is_bucketOrder_dumped_ = true;
        }

        for (auto index : bucket_allreduce_order_) {
          mark_bucket_ready_once(static_cast<size_t>(index));
        }
        
        bucket_allreduce_order_.clear();
      }
    }

    // See Note [Skip allreducing local_used_maps_dev]
    assert(find_unused_parameters_ == false);
    if (find_unused_parameters_) {
      // H2D from local_used_maps_ to local_used_maps_dev_
      for (size_t i = 0; i < local_used_maps_.size(); i++) {
        // We do async H2D to avoid the blocking overhead. The async copy and
        // allreduce respect the current stream, so will be sequenced correctly.
        local_used_maps_dev_[i].copy_(local_used_maps_[i], true);
      }
      local_used_work_ = process_group_comp_->allreduce(local_used_maps_dev_);
    }

    // The autograd engine uses the default stream when running callbacks, so we
    // pass in the current CUDA stream in case it is not the default.
    c10::DeviceType deviceType = replica.contents.device().type();
    const c10::impl::VirtualGuardImpl guard =
        c10::impl::VirtualGuardImpl{deviceType};
    const c10::Stream currentStream =
        guard.getStream(replica.contents.device());
    torch::autograd::Engine::get_default_engine().queue_callback([=] {
      std::lock_guard<std::mutex> lock(this->mutex_);
      // Run callback with the current stream
      c10::OptionalStreamGuard currentStreamGuard{currentStream};
      this->finalize_backward();
    });

    // TODO(Mingzhen) The following is only called when processing the last micro_batch,
    // however, should be called at other micro_batches.
    //if (is_computation_process_) {
    //  auto shm_array=shm_array_.accessor<int, 2>();
    //  shm_array[cntMicroBatch_][0] = ASSIGN_GRADS;
    //}
  }
}

void ElasticReducer::mark_bucket_ready_once(size_t bucket_index) {
    auto& bucket = buckets_[bucket_index];
    std::vector<at::Tensor> tensors;
    tensors.reserve(bucket.replicas.size());
    TORCH_INTERNAL_ASSERT(bucket.replicas.size() == 1);
    tensors.push_back(bucket.replicas[0].contents);

    if (comm_hook_ == nullptr) {
      if (replicas_[0][0].is_cuda() && !shm_grads_[0][0].is_cuda() && (process_group_->getSize() > 1)) {
        bucket.replicas[0].contents_cpu.copy_(bucket.replicas[0].contents, 
                                              /*non_blocking=*/false);
        tensors.clear();
        tensors.push_back(bucket.replicas[0].contents_cpu);
      }
      bucket.work = process_group_->allreduce(tensors);
    } else {
      GradBucket grad_bucket(
          next_bucket_,
          tensors,
          // Since currently we do not support single-process multiple-device
          // mode, we can assume only one replica in the bucket.
          bucket.replicas[0].offsets,
          bucket.replicas[0].lengths,
          bucket.replicas[0].sizes_vec);
      bucket.future_work = comm_hook_->runHook(grad_bucket);
    }
}

// Called when the bucket at the specified index is ready to be reduced.
void ElasticReducer::mark_bucket_ready(size_t bucket_index) {

  TORCH_INTERNAL_ASSERT(bucket_index >= next_bucket_);

  // Buckets are reduced in sequence. Ignore this bucket if
  // it's not its turn to be reduced.
  if (bucket_index > next_bucket_) {
    return;
  }

  // Keep going, until we either:
  // - have kicked off reduction for all buckets, or
  // - found a bucket that's not yet ready for reduction.
  for (; next_bucket_ < buckets_.size() && buckets_[next_bucket_].pending == 0;
       next_bucket_++) {
    auto& bucket = buckets_[next_bucket_];
    std::vector<at::Tensor> tensors;
    tensors.reserve(bucket.replicas.size());
    TORCH_INTERNAL_ASSERT(bucket.replicas.size() == 1);
    tensors.push_back(bucket.replicas[0].contents);
    
    // See Note [DDP Communication Hook]
    // TODO(@sinannasir): merge `work` and `future_work`. Related to GH Issue
    // #41266.
    if (comm_hook_ == nullptr) {
      if ( (!is_bucketOrder_dumped_) && (process_group_->getSize() > 1) ) {
        std::lock_guard<std::mutex> lockGuard(order_mutex_);
        append_allreduce_order(next_bucket_);
      } else {
        if (replicas_[0][0].is_cuda() && !shm_grads_[0][0].is_cuda() && (process_group_->getSize() > 1)) {
          bucket.replicas[0].contents_cpu.copy_(bucket.replicas[0].contents,
                                                /*non_blocking=*/false);
          tensors.clear();
          tensors.push_back(bucket.replicas[0].contents_cpu);
        }
        bucket.work = process_group_->allreduce(tensors);
      }
    } else {
      GradBucket grad_bucket(
          next_bucket_,
          tensors,
          // Since currently we do not support single-process multiple-device
          // mode, we can assume only one replica in the bucket.
          bucket.replicas[0].offsets,
          bucket.replicas[0].lengths,
          bucket.replicas[0].sizes_vec);
      bucket.future_work = comm_hook_->runHook(grad_bucket);
    }
  }
}

void ElasticReducer::initialize_buckets(
    std::vector<std::vector<size_t>> bucket_indices) {
  // If initialize_buckets is called inside DDP constructor, then
  // it does not matter rpc context ptr is nullptr or not, as grad
  // will not be mutated.
  // If initialize_buckets is called during training loop, e.g, inside
  // rebuild_buckets(), since grad could be mutated and be pointed to
  // bucket_view, then it needs to check rpc context ptr is nullptr or not,
  // If rpc context ptr is nullptr, mutate variable.grad(); otherwise,
  // mutate grad in rpc context.
#ifndef _WIN32
  using torch::distributed::autograd::ThreadLocalDistAutogradContext;
  this->rpc_context_.set(ThreadLocalDistAutogradContext::getContextPtr());
#endif

  // This shouldn't be called if we're expecting autograd hooks to fire.
  TORCH_CHECK(
      !expect_autograd_hooks_,
      "`initialize_buckets` must NOT be called during autograd execution.");

  // Clear current bucket assignment.
  buckets_.clear();
  variable_locators_.clear();

  // Ensure we have a bucket index for every variable.
  variable_locators_.resize(replicas_[0].size());

  // Iterate over buckets.
  const auto bucket_count = bucket_indices.size();
  const auto replica_count = replicas_.size();
  buckets_.reserve(bucket_count);
  for (size_t bucket_index = 0; bucket_index < bucket_count; bucket_index++) {
    Bucket bucket;

    // TODO(@pietern): Validate indices.
    // Must be non-empty, unique, and unique across buckets.
    TORCH_CHECK(
        bucket_indices[bucket_index].size() > 0, "Empty bucket specified.");

    // Variables that expect sparse gradients must have their own bucket.
    if (bucket_indices[bucket_index].size() == 1) {
      const auto variable_index = bucket_indices[bucket_index].front();
      bucket.expect_sparse_gradient =
          expect_sparse_gradients_[0][variable_index];
    } else {
      for (const auto variable_index : bucket_indices[bucket_index]) {
        TORCH_CHECK(
            !expect_sparse_gradients_[0][variable_index],
            "Buckets with more than one variable cannot include variables ",
            "that expect a sparse gradient.");
      }
    }

    // Iterate over model replicas.
    for (size_t replica_index = 0; replica_index < replica_count;
         replica_index++) {
      BucketReplica replica;

      if (bucket.expect_sparse_gradient) {
        const auto variable_index = bucket_indices[bucket_index].front();
        const auto& variable = replicas_[replica_index][variable_index];
        TORCH_INTERNAL_ASSERT(bucket_indices[bucket_index].size() == 1);
        replica.variables = {variable};
      } else {
        at::TensorOptions options;
        // The start index of the variable in the flattened tensor.
        size_t offset = 0;

        // Reserve enough space for the per-variable fields stored in bucket
        // replica for efficiency.
        const size_t num_variables = bucket_indices[bucket_index].size();
        replica.variables.reserve(num_variables);
        replica.offsets.reserve(num_variables);
        replica.lengths.reserve(num_variables);
        replica.sizes_vec.reserve(num_variables);

        // Iterate over bucket variables.
        for (const auto variable_index : bucket_indices[bucket_index]) {
          TORCH_CHECK(
              variable_index < replicas_[replica_index].size(),
              "Out of range variable index specified.");
          const auto& variable = replicas_[replica_index][variable_index];
          if (!options.has_device()) {
            options = options.device(variable.device());
          } else {
            TORCH_CHECK(
                variable.device() == options.device(),
                "All parameters in a bucket must be ",
                "placed on the same device.");
          }
          if (!options.has_dtype()) {
            options = options.dtype(variable.dtype());
          } else {
            TORCH_CHECK(
                variable.dtype() == options.dtype(),
                "All parameters in a bucket must have the same dtype.");
          }
          const auto length = variable.numel();
          replica.variables.push_back(variable);
          replica.offsets.push_back(offset);
          replica.lengths.push_back(length);
          replica.sizes_vec.push_back(variable.sizes());
          offset += length;
        }

        // Allocate bucket contents tensor.
        replica.contents = at::empty({static_cast<long>(offset)}, options);
        if (replicas_[0][0].is_cuda() && !shm_grads_[0][0].is_cuda() && (process_group_->getSize() > 1)) {
          replica.contents_cpu = at::empty({static_cast<long>(offset)}, 
                                      options.device(shm_grads_[0][0].device())
                                      ).pin_memory();
        }

        // Note:  "Gradient Layout Contract"
        //
        // Here, create views into the contents tensor for each variable's grad.
        // Views serve as entry points to copy_ each grad's data in/out of the
        // flat contents tensor.
        //
        // Gradients may have dense memory but non-row-major-contiguous strides
        // (e.g. channels_last or channels_last_3d). For coalesced accesses
        // during copy_s, it's beneficial for each view's layout to match its
        // grad's layout.
        //
        // Specifically, we expect torch/csrc/autograd/AccumulateGrad.h produces
        // grads that obey there "Gradient Layout Contract":
        //   (1) if variable.is_non_overlapping_and_dense(), the stashed grad's
        //       strides match variable.
        //   (2) else, stashed grad is rowmajor contiguous.
        // and create views to match.
        //
        // If AccumulateGrad breaks the contract, and produces a grad with an
        // unexpected layout, performance will degrade due to poor memory access
        // patterns when copy_ing grad data in and out of its bucket view.
        // However, numerics remain correct, because the bucket view is the same
        // on either end of the raw allreduce.  bucket_view_in.copy(grad)
        // tranposes
        // (+ densifies) to the bucket view's layout, the data is allreduced,
        // then grad.copy_(bucket_view_out) transposes it back to grad's layout.
        //
        // The only way the numerics can go haywire is if the bucket views
        // themselves have different layouts across processes (or replicas).
        // Bucket views' sizes and strides are set based on param layouts, using
        // the same logic that (we expect) AccumulateGrad uses for their grads.
        // Therefore, the only way a bucket view could have different layouts in
        // different processes is if its param has a different layout in
        // different processes. We can check that param layouts match across
        // processes and replicas in ElasticReducer's constructor by allreducing some
        // metadata.  Checking just once won't catch if someone messes with
        // param layouts over time, but not messing with params after DDP
        // construction is already a documented constraint.
        initialize_bucket_views(replica, replica.contents);
      }

      // Add bucket replica to enclosing bucket.
      bucket.replicas.push_back(std::move(replica));
    }

    // Map participating variables to this bucket.
    // This is identical across replicas so we only need to do this once.
    size_t intra_bucket_index = 0;
    for (const auto variable_index : bucket_indices[bucket_index]) {
      TORCH_CHECK(
          variable_index < variable_locators_.size(),
          "Out of range variable index specified.");
      variable_locators_[variable_index] =
          VariableLocator(bucket_index, intra_bucket_index++);
    }
    bucket.variable_indices = std::move(bucket_indices[bucket_index]);

    buckets_.push_back(std::move(bucket));
  }
}

// (see Note:  "Gradient Layout Contract" in initialize_buckets).
void ElasticReducer::initialize_bucket_views(
    ElasticReducer::BucketReplica& replica,
    at::Tensor& contents) {
  for (size_t i = 0; i < replica.variables.size(); i++) {
    auto& v = replica.variables[i];
    const auto offset = replica.offsets[i];
    const auto length = replica.lengths[i];
    if (v.is_non_overlapping_and_dense()) {
      // If the param's memory is dense, match its layout, anticipating
      // the autograd engine (AccumulateGrad) will also create gradients
      // matching its layout.
      replica.bucket_views_in.push_back(
          contents.as_strided(v.sizes(), v.strides(), offset));
    } else {
      // Fall back to a C-style contiguous view, again anticipating
      // AccumulateGrad will do the same when stashing grads for non-dense
      // params.
      replica.bucket_views_in.push_back(
          contents.narrow(0, offset, length).view(v.sizes()));
    }
    // By default `bucket_views_out` and `bucket_views_in` are
    // essentially the same thing.
    replica.bucket_views_out = replica.bucket_views_in;

    // If gradient_as_bucket_view_ is set as true, then there are two cases to
    // handle: initialize_bucket_views could be called inside initialize_buckets
    // when rebuild_buckets, if grad has already been defined/calculated in
    // previous iteration, old grad needs to be copied into new bucket_view and
    // let grad point to the new bucket_view, initialize_bucket_views could also
    // be called inside initialize_buckets during construction. Grads are not
    // defined during construction time, in this case, do not let grad point to
    // bucket_view, because grads should be kept as being undefined for globally
    // unused parameters.
    if (gradient_as_bucket_view_) {
      auto& bucket_view = replica.bucket_views_in.back();
      runGradCallbackForVariable(v, [&](auto& grad) {
        if (grad.defined() && !grad.is_alias_of(bucket_view)) {
          bucket_view.copy_(grad);
          grad = bucket_view;
          // The grad is modefied and needs to be written back.
          return true;
        }
        // The grad is not modified and does not need to be written back.
        return false;
      });
    }
  }
}

// (see Note:  "Gradient Layout Contract" in initialize_buckets).
void ElasticReducer::populate_bucket_views_out(
    ElasticReducer::BucketReplica& replica,
    at::Tensor& tensor) {
  replica.bucket_views_out.clear();
  for (size_t i = 0; i < replica.variables.size(); i++) {
    const auto& v = replica.variables[i];
    const auto offset = replica.offsets[i];
    const auto length = replica.lengths[i];
    if (v.is_non_overlapping_and_dense()) {
      // If the param's memory is dense, match its layout, anticipating
      // the autograd engine (AccumulateGrad) will also create gradients
      // matching its layout.
      replica.bucket_views_out.push_back(
          tensor.as_strided(v.sizes(), v.strides(), offset));
    } else {
      // Fall back to a C-style contiguous view, again anticipating
      // AccumulateGrad will do the same when stashing grads for non-dense
      // params.
      replica.bucket_views_out.push_back(
          tensor.narrow(0, offset, length).view(v.sizes()));
    }
  }
}

// Traverse the autograd graph starting at the specified output.
// All parameters for which we have a pointer to their gradient accumulation
// functions, but don't show up in the autograd graph will be marked ready for
// for reduction as soon as the first autograd hook is called. This is not
// done immediately because the model output may be ignored, and we only
// want to start performing reductions on `torch.autograd.backward()`.
void ElasticReducer::prepare_for_backward(
    const std::vector<torch::autograd::Variable>& outputs) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::unordered_set<torch::autograd::Node*> seen;
  std::vector<torch::autograd::Node*> queue;

  // Reset accounting.
  expect_autograd_hooks_ = true;
  next_bucket_ = 0;
  backward_stats_base_ = current_time_in_nanos();
  for (auto& bucket : buckets_) {
    for (auto& replica : bucket.replicas) {
      replica.pending = replica.variables.size();
    }
    bucket.pending = bucket.replicas.size();
  }

  // Reset unused parameter accounting.
  has_marked_unused_parameters_ = false;
  unused_parameters_.clear();

  // If find_unused_parameters_ is false, we assume that autograd hooks for ALL
  // variables will be called, and we don't have to search the autograd graph
  // for presence of these hooks.
  if (!find_unused_parameters_) {
    return;
  }

  // Seed queue with the grad functions of all outputs.
  for (const auto& output : outputs) {
    const auto& grad_fn = output.grad_fn();
    if (grad_fn) {
      queue.push_back(grad_fn.get());
    }
  }

  // Traverse the autograd graph starting at the specified output.
  while (!queue.empty()) {
    auto fn = queue.back();
    queue.pop_back();
    for (const auto& edge : fn->next_edges()) {
      if (auto next_ptr = edge.function.get()) {
        const bool was_inserted = seen.insert(next_ptr).second;
        if (was_inserted) {
          queue.push_back(next_ptr);
        }
      }
    }
  }

  // Find accumulator functions that don't show up in this graph.
  for (const auto& it : gradAccToVariablesMap_) {
    // If the accumulator function is present in the graph, we know
    // a gradient will be computed for the corresponding parameter.
    if (seen.count(it.first) == 0) {
      auto& indices = it.second;
      unused_parameters_.reserve(unused_parameters_.size() + indices.size());
      unused_parameters_.insert(
          unused_parameters_.end(), indices.begin(), indices.end());
    }
  }

  // Warn user about unnecessary perf hit if all parameters were used.
  if (unused_parameters_.empty()) {
    TORCH_WARN_ONCE(
        "find_unused_parameters=True was specified in DDP constructor, "
        "but did not find any unused parameters. This flag results in an extra "
        "traversal of the autograd graph every iteration, which can adversely "
        "affect performance. If your model indeed never has any unused "
        "parameters, consider turning this flag off. Note that this warning may "
        "be a false positive your model has flow control causing later iterations "
        "to have unused parameters.");
  }
}

void ElasticReducer::copy_bucket_to_grad(
    torch::autograd::Variable& variable,
    ElasticReducer::BucketReplica& replica,
    size_t intra_bucket_index,
    bool global_unused) {
  const auto& bucket_view = replica.bucket_views_out[intra_bucket_index];
  runGradCallbackForVariable(variable, [&](auto& grad) {
    // If a parameter is globally unused, we keep its grad untouched.
    if (!global_unused) {
      if (!grad.defined()) {
        // Creates grad according to the "Gradient Layout Contract"
        // (see torch/csrc/grad/AccumulateGrad.h)
        grad =
            torch::autograd::utils::clone_obey_contract(bucket_view, variable);
      } else {
        grad.copy_(bucket_view);
      }
      // The grad is modified and needs to be written back.
      return true;
    }
    // The grad is not modified.
    return false;
  });
}

// A bucket with one or more dense tensors needs to be unflattened.
void ElasticReducer::finalize_bucket_dense(Bucket& bucket) {
  for (size_t replica_index = 0; replica_index < bucket.replicas.size();
       replica_index++) {
    auto& replica = bucket.replicas[replica_index];
    for (size_t intra_bucket_index = 0;
         intra_bucket_index < replica.variables.size();
         intra_bucket_index++) {
      auto& variable = replica.variables[intra_bucket_index];
      const auto offset = replica.offsets[intra_bucket_index];
      const auto length = replica.lengths[intra_bucket_index];

      bool global_unused = false;
      // See Note [Skip allreducing local_used_maps_dev]
      if (find_unused_parameters_) {
        // Determine if this param has been used globally or not.
        //
        // If the variable was used locally, it is also used globally and then
        // we don't need to wait for the reduction. Otherwise we lazily wait for
        // the reduction to complete, only when we see a variable that was
        // unused locally. Then we end up delaying the synchronization point
        // that local_used_work_->wait() implies. If we don't have any unused
        // parameters at all, we can skip waiting for the work to complete
        // altogether, and cause negligible performance overhead for models
        // where all parameters are used. Such lazily waiting means minimizing
        // performance impact for the big majority of models where all
        // parameters are always used. Then we only pay the overhead cost if
        // there is indeed a parameter that is locally unused, because we need
        // to check if it's also globally unused.
        size_t variable_index = bucket.variable_indices[intra_bucket_index];
        // Note: global_unused might not be global yet. As we lazily wait for
        // the reduction to complete, it becomes really global only if we get to
        // the point as below where we wait for the reduction work, make D2H
        // copy, and update global_unused with the real global consensus, i.e.
        // local_used_maps_reduced_ is true.
        global_unused =
            local_used_maps_[replica_index][variable_index].item<int>() == 0;
        if (global_unused && !local_used_maps_reduced_) {
          // Wait for local_used_maps reduction to complete.
          local_used_work_->wait();
          // D2H from local_used_maps_dev_ to local_used_maps_
          for (size_t i = 0; i < local_used_maps_.size(); i++) {
            local_used_maps_[i].copy_(local_used_maps_dev_[i]);
          }
          global_unused =
              local_used_maps_[replica_index][variable_index].item<int>() == 0;
          local_used_maps_reduced_ = true;
        }
      }

      if (!gradient_as_bucket_view_) {
        copy_bucket_to_grad(
            variable, replica, intra_bucket_index, global_unused);
      } else {
        const auto& bucket_view_out =
            replica.bucket_views_out[intra_bucket_index];
        auto& bucket_view_in = replica.bucket_views_in[intra_bucket_index];
        // If communication_hook is registered, bucket_view_out stores
        // allreduced results in a newly allocated tensor, copy bucket_view_out
        // back to bucket_view_in that referring to replica.content tensor and
        // grad.
        if (!bucket_view_in.is_alias_of(bucket_view_out)) {
          bucket_view_in.copy_(bucket_view_out);
        }
        runGradCallbackForVariable(variable, [&](auto& grad) {
          // If a parameter is globally unused, we keep its grad untouched.
          if (!global_unused) {
            // If grad is globally used but locally unused, let grad point to
            // bucket_view_in
            if (!grad.defined()) {
              grad = bucket_view_in;
            } else {
              if (!grad.is_alias_of(bucket_view_in)) {
                grad.copy_(bucket_view_in);
                TORCH_WARN_ONCE(
                    "Detected at least one parameter gradient is not the "
                    "expected DDP bucket view when setting "
                    "gradient_as_bucket_view=True. This can happen when "
                    "multiple parameters sharing the same gradient. For "
                    "example, param0 and param1 share the same gradient "
                    "grad0. In this case, grad0 would first point to "
                    "bucket_view_in0 when param0 is ready. Later, when "
                    "param1 is ready, it will override grad0 to point to "
                    "bucket_view_in1. However, param0 still expects grad0 "
                    "to point to bucket_view_in0, and hence hit this "
                    "warning. If you saw this message, please double-check if "
                    "the above situation is expected for your application.");
              }
            }
            // The grad is modified and needs to be written back.
            return true;
          }
          // The grad is not modified.
          return false;
        });
      }
    }
  }
}

void ElasticReducer::finalize_backward() {
  // No longer expect autograd hooks to fire after this function returns.
  TORCH_INTERNAL_ASSERT(expect_autograd_hooks_);
  expect_autograd_hooks_ = false;

  // No longer require call to finalize after this function returns.
  TORCH_INTERNAL_ASSERT(require_finalize_);
  require_finalize_ = false;

  // Unset allreduce division factor, as it may change in next backwards pass
  // when running with DDP join mode.
  divFactor_ = kUnsetDivFactor;

  // Check that all buckets were completed and had their work kicked off.
  TORCH_INTERNAL_ASSERT(next_bucket_ == buckets_.size());

  // Wait for asynchronous reduction to complete and unflatten contents.
  for (auto& bucket : buckets_) {
    // See Note [DDP Communication Hook]
    if (comm_hook_ == nullptr) {
      TORCH_INTERNAL_ASSERT(
          bucket.work,
          "Expected bucket.work not to be null. "
          "This may indicate that allreduce hooks were not properly installed.");
      bucket.work->wait();
      
      if (replicas_[0][0].is_cuda() && !shm_grads_[0][0].is_cuda() && (process_group_->getSize() > 1)) {
        bucket.replicas[0].contents.copy_(bucket.replicas[0].contents_cpu,
                                          /*non_blocking=*/false);
      }

    } else {
      TORCH_INTERNAL_ASSERT(
          bucket.future_work,
          "Expected bucket.future_work not to be null. "
          "This may indicate that communication hook was not properly installed.");
      bucket.future_work->wait();

      auto future_result =
          comm_hook_->parseHookResult(bucket.future_work->value());

      for (size_t i = 0; i < future_result.size(); i++) {
        auto& replica = bucket.replicas[i];
        if (bucket.expect_sparse_gradient) {
          replica.contents.copy_(future_result[i]);
        } else {
          // Reinitialize only `bucket_views_out` with the future_result by
          // following the same logic in `initialize_buckets`.
          populate_bucket_views_out(replica, future_result[i]);
        }
      }
    }
    if (!bucket.expect_sparse_gradient) {
      // We don't need to finalize the sparse bucket since the sparse grad and
      // the bucket essentially point to the same storage. As a result, once
      // the allreduce is done, the sparse grads are automatically updated.
      finalize_bucket_dense(bucket);
    }
  }

  // See Note [Skip allreducing local_used_maps_dev]
  if (find_unused_parameters_) {
    // Reset unused parameter accounting.
    for (auto& local_used : local_used_maps_) {
      local_used.fill_(0);
    }
    // Due to the lazy wait, it is possible that reduction of the current
    // iteration is still going when the one for next iteration gets kicked off.
    // For such case, we want to wait explicitly to make sure the reduction does
    // complete before kicking off next one. Otherwise the previous one may
    // interfere, write to the device-side memory and clobber the content of
    // local_unused_maps_dev_.
    if (!local_used_maps_reduced_) {
      local_used_work_->wait();
    }
    local_used_maps_reduced_ = false;
  }

  if (!is_rebuilt_param_indices_saved_) {
    assert(!rebuilt_param_indices_.empty());
    rebuilt_param_indices_backup_.assign(rebuilt_param_indices_.begin(), 
                                        rebuilt_param_indices_.end());
    is_rebuilt_param_indices_saved_ = true;   
  }
}

void ElasticReducer::runGradCallbackForVariable(
    torch::autograd::Variable& variable,
    GradCallback&& cb) {
  auto context_ptr = rpc_context_.context_ptr.load();
  if (context_ptr == nullptr) {
    cb(variable.mutable_grad());
  } else {
    // Under distributed autograd
#ifndef _WIN32
    context_ptr->runGradCallbackForVariable(variable, std::move(cb));
#endif
  }
}

void ElasticReducer::RpcContext::set(ContextPtr&& new_context_ptr) {
  // We should set 'new_context_ptr' even if it's nullptr. That means the
  // reducer is under a local backward run.
  const auto new_context_raw_ptr = new_context_ptr.get();
  if (context_ptr.exchange(new_context_raw_ptr) != new_context_raw_ptr) {
    // Set the shared ptr to the context only if it's set first time.
    // All call sites should use the same context ptr.
    // Use an atomic to avoid data race from multiple threads.
    context_ptr_holder = std::move(new_context_ptr);
  }
}

void ElasticReducer::sync_bucket_indices(
    std::vector<std::vector<size_t>>& bucket_indices) {
  auto num_buckets = bucket_indices.size();
  std::vector<size_t> bucket_sizes;
  bucket_sizes.reserve(num_buckets);
  int64_t total_size = 0;
  for (size_t i = 0; i < num_buckets; i++) {
    auto bucket_size = bucket_indices.at(i).size();
    bucket_sizes.push_back(bucket_size);
    total_size += bucket_size;
  }

  at::TensorOptions options;
  options = options.dtype(at::kInt);
  if (shm_grads_[0][0].is_cuda()) {
    options = options.device(replicas_[0][0].device());
  }

  // Group indices and num_bucket together into indices_tensor
  // Broadcast this tensor first, as its size is equal among all processes
  auto indices_tensor = at::empty({total_size + 1}, at::kInt);
  auto indices_accessor = indices_tensor.accessor<int, 1>();
  auto indices_accessor_Index = 0;
  for (size_t i = 0; i < num_buckets; i++) {
    const auto& bucket_size = bucket_indices.at(i).size();
    for (size_t j = 0; j < bucket_size; j++) {
      indices_accessor[indices_accessor_Index++] = bucket_indices[i][j];
    }
  }
  indices_accessor[indices_accessor_Index] = num_buckets;

  // Copy CPU tensor to device tensor, as the process_group_ could be NCCL and
  // it can only broadcast device tensors.
  auto indices_tensor_device = at::empty({total_size + 1}, options);
  indices_tensor_device.copy_(indices_tensor, /*non_blocking=*/true);
  std::vector<at::Tensor> indices_tensor_list = {indices_tensor_device};
  process_group_->broadcast(indices_tensor_list)->wait();
  indices_tensor.copy_(indices_tensor_list.front(), /*non_blocking=*/false);

  // Update num_buckets after receiving it from rank 0
  num_buckets = indices_accessor[indices_accessor_Index];

  // Broadcast bucket_sizes
  auto bucket_sizes_tensor = at::empty({(int64_t)num_buckets}, at::kInt);
  auto bucket_sizes_accessor = bucket_sizes_tensor.accessor<int, 1>();
  for (size_t i = 0; i < num_buckets; i++) {
    // For rank != 0, it is possible that local num buckets bucket_sizes.size()
    // is smaller than broadcasted num_buckets
    bucket_sizes_accessor[i] =
        bucket_sizes.at(std::min(i, (bucket_sizes.size() - 1)));
  }
  auto bucket_sizes_tensor_device = at::empty({(int64_t)num_buckets}, options);
  bucket_sizes_tensor_device.copy_(bucket_sizes_tensor, /*non_blocking=*/true);
  std::vector<at::Tensor> bucket_sizes_tensor_list = {
      bucket_sizes_tensor_device};
  process_group_->broadcast(bucket_sizes_tensor_list)->wait();
  bucket_sizes_tensor.copy_(
      bucket_sizes_tensor_list.front(), /*non_blocking=*/false);

  // Clear bucket_indices first, and then update bucket_indices using received
  // num_buckets, bucket_sizes_tensor and indices_tensor from rank 0
  bucket_indices.clear();
  bucket_indices.reserve(num_buckets);
  indices_accessor_Index = 0;
  for (size_t i = 0; i < num_buckets; i++) {
    const auto& bucket_size = bucket_sizes_accessor[i];
    std::vector<size_t> bucket;
    bucket.reserve(bucket_size);
    for (size_t j = 0; j < bucket_size; j++) {
      bucket.push_back(indices_accessor[indices_accessor_Index++]);
    }
    bucket_indices.emplace_back(std::move(bucket));
  }
}

bool ElasticReducer::rebuild_buckets() {
  // Ensure reduction for previous backwards pass is finished. If user's model
  // has unused parameters for example, this will raise an error recommending to
  // run with find_unused_parameters=True, instead of the size mismatch
  // exception below.
  if (is_computation_process_) {
    ensure_prior_reduction_finished();

    std::lock_guard<std::mutex> lock(mutex_);
    if ( !should_force_rebuild_buckets_ && \
        (!should_rebuild_buckets() || rebuilt_params_.empty()) ) {
      return false;
    }

    auto shm_array=shm_array_.accessor<int, 2>();
    for (int i = 0; i < logic_gpus_num_ - 1; i++) {
      shm_array[i][0] = REBUILD_BUCKETS;
    }
  }

  process_group_->barrier()->wait();
  
  // broadcast rebuilt_param_indices_ from cheif_rank to all ranks
  customized_broadcast(rebuilt_param_indices_, cheif_rank_);

  if (should_force_rebuild_buckets_ || !is_computation_process_) {
    TORCH_INTERNAL_ASSERT(
        rebuilt_params_.empty(),
        c10::str("rebuild_params_ should be empty"));
    for (auto& variable_index : rebuilt_param_indices_) {
      rebuilt_params_.push_back(replicas_[0][variable_index]);
    }
  }

  TORCH_INTERNAL_ASSERT(
      rebuilt_params_.size() == rebuilt_param_indices_.size(),
      c10::str(
          "rebuilt parameter tensors size is not same as rebuilt parameter indices size: ",
          rebuilt_params_.size(),
          " versus ",
          rebuilt_param_indices_.size()));
  TORCH_INTERNAL_ASSERT(
      replicas_[0].size() == rebuilt_param_indices_.size(),
      c10::str(
          "rebuilt parameter indices size is not same as original model parameters size.",
          "Original model param size is: ",
          replicas_[0].size(),
          " versus rebuilt params size of: ",
          rebuilt_param_indices_.size()));
  std::vector<std::vector<size_t>> rebuilt_bucket_indices;
  std::vector<size_t> bucket_size_limits;
  bucket_size_limits.push_back(kDefaultFirstBucketBytes);
  bucket_size_limits.push_back(bucket_bytes_cap_);
  rebuilt_bucket_indices = compute_bucket_assignment_by_size(
      rebuilt_params_,
      bucket_size_limits,
      expect_sparse_gradients_[0],
      rebuilt_param_indices_);

  // For rebuilt bucket indices, it needs to be synced across all ranks.
  // Broadcast the newly rebuilt bucket indices from rank 0 in default.
  // After syncing up rebuilt bucket indices, initialize buckets for reducer.
  sync_bucket_indices(rebuilt_bucket_indices);

  has_rebuilt_bucket_ = true;
  rebuilt_params_.clear();
  rebuilt_param_indices_.clear();

  initialize_buckets(std::move(rebuilt_bucket_indices));
  return true;
}

// See Note [DDP Communication Hook]
void ElasticReducer::register_comm_hook(std::unique_ptr<CommHookInterface> iface) {
  TORCH_CHECK(
      comm_hook_ == nullptr,
      "register_comm_hook or register_builtin_comm_hook can only be called once.");
  // TODO(#42542): Single-process multiple-device mode support for DDP
  // communication hook.
  TORCH_CHECK(
      replicas_.size() == 1,
      "Communication hook does not support single-process multiple-device mode.");

  comm_hook_ = std::move(iface);
}

// See Note [DDP Communication Hook]
void ElasticReducer::register_builtin_comm_hook(
    c10d::BuiltinCommHookType comm_hook_type) {
  TORCH_CHECK(
      comm_hook_ == nullptr,
      "register_builtin_comm_hook or register_comm_hook can only be called once.");
  TORCH_CHECK(
      replicas_.size() == 1,
      "Communication hook does not support single-process multiple-device mode.");
  // TODO: Support GLOO and MPI backends for DDP communication hook.
  TORCH_CHECK(
      process_group_->getBackendName() == "nccl",
      "register_builtin_comm_hook currently can only support NCCL backend, but the current backend is %s.",
      process_group_->getBackendName());

  switch (comm_hook_type) {
    case c10d::BuiltinCommHookType::ALLREDUCE:
      comm_hook_ =
          std::make_unique<c10d::AllReduceCommHook>(process_group_.get());
      LOG(INFO) << "Built-in communication hook ALLREDUCE is registered.";
      break;
    case c10d::BuiltinCommHookType::FP16_COMPRESS:
      comm_hook_ =
          std::make_unique<c10d::FP16CompressCommHook>(process_group_.get());
      LOG(INFO) << "Built-in communication hook FP16_COMPRESS is registered.";
      break;
    default:
      TORCH_WARN_ONCE(
          "Unknown built-in DDP comm hook type is provided. No comm hook will be used.");
  }
}

void ElasticReducer::ensure_prior_reduction_finished() {
  // Check that any prior reduction has finished.
  // The variable `require_finalize_` is true until all gradients
  // have been computed and reduction of all buckets has been kicked off.
  if (require_finalize_) {
    TORCH_CHECK(
        false,
        "Expected to have finished reduction in the prior iteration before ",
        "starting a new one. ",
        "",
        "This error indicates that your module has parameters that were ",
        "not used in producing loss. ",
        "",
        "You can enable unused parameter detection by (1) passing the keyword "
        "argument `find_unused_parameters=True` to ",
        "`torch.nn.parallel.DistributedDataParallel`; (2) making sure all ",
        "`forward` function outputs participate in calculating loss. "
        "",
        "If you already have done the above two steps, then the distributed ",
        "data parallel module wasn't able to locate the output tensors in the ",
        "return value of your module's `forward` function. ",
        "Please include the loss function and the structure of the return ",
        "value of `forward` of your module when reporting this issue (e.g. ",
        "list, dict, iterable).");
  }
}

void ElasticReducer::copy_grad_to_bucket_total_commProcess() {
  for (auto& bucket : buckets_ ) {
    assert(bucket.replicas.size() == 1);
    auto& rep = bucket.replicas[0];

    for (const auto variable_index : bucket.variable_indices) {
      const auto& variable = replicas_[0][variable_index];
      const auto& variable_locator = variable_locators_[variable_index];
      const size_t intra_bucket_index = variable_locator.intra_bucket_index;
      auto& bucket_view = rep.bucket_views_in[intra_bucket_index];

      // TODO(Mingzhen) Why should we divide divFactor_ here?
      copy_grad_to_bucket(variable.grad(), bucket_view);

    }
  }
}

void ElasticReducer::copy_bucket_to_grad_total_commProcess() {
  for (auto& bucket : buckets_ ) {
    assert(bucket.replicas.size() == 1);
    auto& rep = bucket.replicas[0];

    for (const auto variable_index : bucket.variable_indices) {
      auto& variable = replicas_[0][variable_index];
      const auto& variable_locator = variable_locators_[variable_index];
      const size_t intra_bucket_index = variable_locator.intra_bucket_index;
      auto& bucket_view = rep.bucket_views_out[intra_bucket_index];

      // TODO(Mingzhen) set global_unused to false in POC
      bool global_unused = false;
      copy_bucket_to_grad(variable.mutable_grad(), rep, intra_bucket_index, global_unused);

    }
  }
}

bool ElasticReducer::wait_for_shm_signal(const int signal, const int pos) const {
  constexpr int _10US = 10;
  auto shm_array=shm_array_.accessor<int,2>();
  while(shm_array[local_rank_][pos] != signal) {
    usleep(_10US);
  }
  return true;
}

void ElasticReducer::do_allreduce_commProcess() {
      auto& grad_local = shm_grads_;
      if (!is_grad_assigned_) {
        size_t it = 0;
        for (auto param : replicas_[0]) {
          param.mutable_grad() = grad_local[it].detach();
          it += 1;
        }
        assert(it == replicas_[0].size());
        is_grad_assigned_ = true;
      }

      copy_grad_to_bucket_total_commProcess();
      
      if (!is_bucketOrder_dumped_) {
        wait_for_shm_signal(ORDER_DUMPED, 1);
        customized_broadcast(bucket_allreduce_order_, cheif_rank_);
        cnt_bucketOrder_ += 1;
        if (cnt_bucketOrder_ >= 2) {
          is_bucketOrder_dumped_ = true;
        }
      }

      for (size_t bucket_index : bucket_allreduce_order_) {
        auto& bucket = buckets_[bucket_index];
        std::vector<at::Tensor> tensors;
        tensors.reserve(bucket.replicas.size());
        assert(bucket.replicas.size() == 1);
        tensors.push_back(bucket.replicas[0].contents);

        // The tensors can be CPU tensor or GPU tensor.
        // Depend on where the model locates.
        bucket.work = process_group_->allreduce(tensors);
      }

      for (size_t bucket_index : bucket_allreduce_order_) {
        auto& bucket = buckets_[bucket_index];

        TORCH_INTERNAL_ASSERT(
            bucket.work,
            "Expected bucket.work not to be null. "
            "This may indicate that allreduce hooks were not properly installed.");
        bucket.work->wait();
      }
      // The model gradients of communication processes do not need updating, 
      // as the gradients are never used.
      // copy_bucket_to_grad_total_commProcess();
}

// Core of the ElasticReducer
void ElasticReducer::commProcess_listening() {
  TORCH_CHECK(is_computation_process_ == false, "commProcess_listening() should only be called by commProcess.");

  if (logic_gpus_num_ < 2) return;

  if (divFactor_ == kUnsetDivFactor) {
    divFactor_ = process_group_->getSize();
  }
  
  int now_value;
  int pre_value = 0;

  auto shm_array=shm_array_.accessor<int, 2>();

  constexpr int _10US = 10;

  while(true) {
    // block to recv new update msg
    while(true) {
      usleep(_10US);
      
      if (shm_array[local_rank_][0] != pre_value) {
        now_value = shm_array[local_rank_][0];
        pre_value = now_value;
        shm_array[local_rank_][0] = WAITING;
        break;
      }
    }

    switch (now_value) {
      case WAITING:
        // waiting
        break;
      case ASSIGN_GRADS:
        do_allreduce_commProcess();
        pre_value = ASSIGN_GRADS_FINISH;
        break;
      case REBUILD_BUCKETS:
        rebuild_buckets();
        pre_value = REBUILD_BUCKETS_FINISH;
        break;
      case JOB_FINISH:
        return;
      default:
        assert(0 && "Illegal shm signal");
        return;
    }

  }
}

void ElasticReducer::customized_broadcast(std::vector<int64_t>& data_vec, const int src_rank ) {
  const bool is_src = (process_group_->getRank() == src_rank);
  
  at::TensorOptions tensor_opts;
  tensor_opts = tensor_opts.dtype(torch::kInt64);
  if (shm_grads_[0][0].is_cuda()) {
    tensor_opts = tensor_opts.device(replicas_[0][0].device()); 
  }

  ::c10d::BroadcastOptions bcast_opts;
  bcast_opts.rootRank = src_rank;
  bcast_opts.rootTensor = 0;

  // broadcast len(data_vec) from src_rank
  int64_t len = 0;
  if (is_src) {
    len = static_cast<int64_t>(data_vec.size());}
  auto len_tensor = at::zeros({1}, torch::kInt64) + len;
  auto len_accessor = len_tensor.accessor<int64_t, 1>();
  const int len_accessor_Index = 0;
  auto len_tensor_device = at::empty({1}, tensor_opts);
  len_tensor_device.copy_(len_tensor, /*non_blocking=*/false);
  std::vector<at::Tensor> len_tensor_list = {len_tensor_device};
  process_group_->broadcast(len_tensor_list, bcast_opts)->wait();
  len_tensor.copy_(len_tensor_list.front(), /*non_blocking=*/false);
  len = len_accessor[len_accessor_Index];
  
  // broadcast data_vec from src_rank
  at::Tensor data_tensor;
  if (is_src) {
    data_tensor = torch::from_blob((int64_t*)(data_vec.data()), data_vec.size(), torch::kInt64).clone();}
  else {
    data_tensor = at::zeros({len}, torch::kInt64);}
  auto data_tensor_device = at::empty({len}, tensor_opts);
  data_tensor_device.copy_(data_tensor, /*non_blocking=*/false);
  auto data_accessor = data_tensor.accessor<int64_t, 1>();
  std::vector<at::Tensor> data_tensor_list = {data_tensor_device};
  process_group_->broadcast(data_tensor_list, bcast_opts)->wait();
  data_tensor.copy_(data_tensor_list.front(), /*non_blocking=*/false);

  data_vec.clear();
  for (int64_t i = 0; i < len; i++) {
    data_vec.push_back(data_accessor[i]);
  }  
}

void ElasticReducer::append_allreduce_order(const int64_t bucket_index) {
  bucket_allreduce_order_.push_back(bucket_index);
}

at::Tensor ElasticReducer::get_rebuilt_param_indices() {
  assert(is_rebuilt_param_indices_saved_ && !rebuilt_param_indices_backup_.empty());
  at::TensorOptions tensor_opts;
  tensor_opts = tensor_opts.dtype(torch::kInt64);

  at::Tensor data_tensor;
  data_tensor = torch::from_blob((int64_t*)(rebuilt_param_indices_backup_.data()), rebuilt_param_indices_backup_.size(), torch::kInt64).clone();
  return data_tensor;
}

void ElasticReducer::force_load_rebuilt_param_indices(at::Tensor indices) {
  auto accessor = indices.accessor<int64_t, 1>();
  rebuilt_params_.clear();
  rebuilt_param_indices_.clear();
  rebuilt_param_indices_backup_.clear();
  for (int64_t i = 0; i < indices.sizes()[0]; i++) {
    rebuilt_param_indices_.push_back(accessor[i]);
    rebuilt_param_indices_backup_.push_back(accessor[i]);
  }
  is_rebuilt_param_indices_saved_ = true;
}

void ElasticReducer::force_rebuild_buckets() {
  assert(is_rebuilt_param_indices_saved_ && 
        "Should force_load_rebuilt_param_indices before force_rebuild_buckets.");
  should_force_rebuild_buckets_ = true;
  rebuild_buckets();
  should_force_rebuild_buckets_ = false;
  is_force_rebuild_buckets_finished_ = true;
}

void ElasticReducer::set_construction_logging_data(
    const std::string& module_name,
    const std::vector<int>& device_ids,
    int output_device,
    bool broadcast_buffers) {
  ddp_logging_data_->module_name = module_name;
  ddp_logging_data_->device_ids = device_ids;
  ddp_logging_data_->output_device = output_device;
  ddp_logging_data_->broadcast_buffers = broadcast_buffers;
  ddp_logging_data_->world_size = process_group_->getSize();
  ddp_logging_data_->rank = process_group_->getRank();
  ddp_logging_data_->bucket_cap_mb = bucket_bytes_cap_ / (1024 * 1024);
  ddp_logging_data_->find_unused_parameters = find_unused_parameters_;
  ddp_logging_data_->gradient_as_bucket_view = gradient_as_bucket_view_;
  ddp_logging_data_->backend_name = process_group_->getBackendName();

  LogPyTorchDDPUsage(*ddp_logging_data_);
}

c10::DDPLoggingData ElasticReducer::get_ddp_logging_data() {
  return *ddp_logging_data_;
}

namespace {

// Tensors may be coalesced into buckets. Buckets must contain tensors of
// the same type, on the same device, so a bucket can identified by a
// composite key of a tensor's type identifier and its device.
struct BucketKey {
  BucketKey(c10::ScalarType type, c10::Device device)
      : type(std::move(type)), device(std::move(device)) {}

  const c10::ScalarType type;
  const c10::Device device;

  // See torch/csrc/utils/hash.h for dispatch code.
  static size_t hash(const BucketKey& key) {
    return c10::get_hash(key.type, key.device);
  }
};

inline bool operator==(const BucketKey& lhs, const BucketKey& rhs) {
  return lhs.type == rhs.type && lhs.device == rhs.device;
}

} // namespace


} // namespace c10d
