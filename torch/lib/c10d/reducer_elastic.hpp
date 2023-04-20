#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <c10d/comm.hpp>
#include <c10/util/intrusive_ptr.h>
#include <c10d/ProcessGroup.hpp>
#include <c10d/default_comm_hooks.hpp>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/distributed/autograd/context/context.h>

#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>

#define DO_ALLREDUCE  3
#define DO_ALLREDUCE_FINISH  4
#define REBUILD_BUCKETS  5
#define REBUILD_BUCKETS_FINISH  6
#define FINALIZE_BACKWARD  7
#define FINALIZE_BACKWARD_FINISH  8
#define ASSIGN_GRADS 9
#define ASSIGN_GRADS_FINISH 10
#define JOB_FINISH  100
#define WAITING 0
#define ORDER_DUMPED 1000

namespace c10d {

class ElasticReducer {
 public:
  // The constructor takes a list of variables for every model replica.
  // The bucket assignment for this reducer is specified as a list of
  // buckets, each of which is specified as a list of indices into the
  // variables list for **a single replica** (i.e. `variables[0]`).
  explicit ElasticReducer(
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
      std::vector<at::Tensor> shm_grads);

  ~ElasticReducer() noexcept(false);

  // To (re-)initialize bucket assignment, pass a list of buckets, each
  // of which is specified by a list of indices in the variables list.
  // This function performs validation that the variables within a bucket
  // all live on the same device and have the same dimensionality.
  void initialize_buckets(std::vector<std::vector<size_t>> bucket_indices);

  // This function is called when the forward function has produced an output,
  // and the user wishes to reduce gradients in the backwards pass.
  // If they don't, and wish to accumulate gradients before reducing them,
  // a call to this function can simply be omitted.
  void prepare_for_backward(
      const std::vector<torch::autograd::Variable>& outputs);

  // Returns the relative time in nanoseconds when gradients were ready,
  // with respect to the time `prepare_for_backward` was called. The outer
  // vector is for model replicas and the inner vector is for parameters.
  std::vector<std::vector<int64_t>> get_backward_stats() const {
    return backward_stats_;
  }

  // Registers a hook to the reducer. The hook is `CommHookInterface`
  // type to allow both Python and CPP hooks. This function can only
  // be called once before calling backward.
 // Cannot combine with the call of `register_builtin_comm_hook`.
  void register_comm_hook(std::unique_ptr<CommHookInterface> iface);

  // Registers a built-in C++ comm hook to the reducer. This function can only
  // be called once before calling backward.
  // Cannot combine with the call of `register_comm_hook`.
  void register_builtin_comm_hook(c10d::BuiltinCommHookType comm_hook_type);

  // Returns a vector of tensors in each bucket in sequential order.
  std::vector<std::vector<at::Tensor>> get_bucket_tensors() const;

  // Rebuild buckets based on rebuilt_params_ and rebuilt_param_indices_
  // according to when tensors received grads in the backward pass.
  // TODO this function makes broadcast communication call and
  // could be overlapped with next forward() call, thus
  // it could be async. Will make it async when rebuilding buckets for
  // find_unused_parameters = true case, as we could rebuild buckets more than
  // once for find_unused_parameters = true case, where subgraphs are trained
  // and parameter indices order may change more frequently.
  // For find_unused_parameters = false case, buckets are only rebuilt once,
  // the performance cost is negligible. Returns true if the buckets were
  // rebuilt.
  bool rebuild_buckets();

  // Returns true if we should rebuild buckets, else false. We only rebuild
  // buckets once after the first iteration and never rebuild them if
  // find_unused_parameters_.
  inline bool should_rebuild_buckets() const {
    if (is_force_rebuild_buckets_finished_) {
      return false;
    }
    return !find_unused_parameters_ && !has_rebuilt_bucket_;
  }


  // Count the number of calculated mini batch
  inline void count_micro_batch() {
    cntMicroBatch_ += 1;
    //fprintf(stderr, "\033[47;31m## current cntMiniBatch = %ld \033[0m\n", cntMicroBatch_ );
    cntMicroBatch_ = cntMicroBatch_ % logic_gpus_num_;
  }


  // Pushes all parameters to be rebuilt.
  void push_rebuilt_params_for_all_indices();

  // Creates and sets ForwardPassWorkHandle given a ProcessGroup::Work and the
  // corresponding tensor being reduced.
  void set_forward_pass_work_handle(
      c10::intrusive_ptr<c10d::ProcessGroup::Work> forwardPassWorkHandle,
      bool useStaticWorldSize);

  // Retrieve on-device tensors used to track locally unused parameters. For
  // each replica, it is a tensor where index i = 1 if the Variable with that
  // index has been used.
  std::vector<at::Tensor> get_local_used_maps_on_device() const;

  // Set logging data that can be got during DistributedDataParallel
  // construction time.
  void set_construction_logging_data(
      const std::string& module_name,
      const std::vector<int>& device_ids,
      int output_device,
      bool broadcast_buffers);

  // An Interface for users to get DDPLoggingData and log them
  // in the applications.
  c10::DDPLoggingData get_ddp_logging_data();


  // For communication process, listening to the shared memory signal
  void commProcess_listening();

  at::Tensor get_shm_array() {
    return shm_array_;
  }

  void print_shm_array() {
    std::cout << shm_array_ << std::endl;
  }

  at::Tensor get_rebuilt_param_indices();
  void force_load_rebuilt_param_indices(at::Tensor indices);
  void force_rebuild_buckets();

 protected:
  // Forward declaration.
  struct Bucket;
  // Locates a specific variable by replica index and variable index.
  struct VariableIndex {
    size_t replica_index;
    size_t variable_index;

    VariableIndex() = default;

    VariableIndex(size_t replica_index_, size_t variable_index_) {
      replica_index = replica_index_;
      variable_index = variable_index_;
    }
  };

  void push_rebuilt_params(const VariableIndex& index);

  mutable std::mutex mutex_;
  std::vector<std::vector<torch::autograd::Variable>> replicas_;
  c10::intrusive_ptr<::c10d::ProcessGroup> process_group_;
  c10::intrusive_ptr<::c10d::ProcessGroup> process_group_comp_;
  std::vector<std::vector<bool>> expect_sparse_gradients_;

  std::vector<std::vector<std::shared_ptr<torch::autograd::Node>>>
      grad_accumulators_;
  std::unordered_map<torch::autograd::Node*, std::vector<VariableIndex>>
      gradAccToVariablesMap_;
  std::vector<std::pair<uintptr_t, std::shared_ptr<torch::autograd::Node>>>
      hooks_;

  bool expect_autograd_hooks_;
  bool require_finalize_;
  size_t next_bucket_;

  bool has_marked_unused_parameters_;
  const bool find_unused_parameters_;
  const bool gradient_as_bucket_view_;
  std::vector<VariableIndex> unused_parameters_;
  // Locally used parameter maps indicating if parameters are used locally
  // during the current iteration or no_sync session if no_sync is on. One
  // tensor for each model replica and each tensor is one-dim int32 tensor of
  // number of parameters. These tensors are marked in autograd_hook to indicate
  // the corresponding param has been used, and get allreduced in the end of
  // backward of current iteration or no_sync session for figuring out the
  // globally unused parameters.
  //
  // local_used_maps_:     CPU tensors for bookkeeping locally used params
  // local_used_maps_dev_: dev tensors for reducing globally unused params
  std::vector<at::Tensor> local_used_maps_;
  std::vector<at::Tensor> local_used_maps_dev_;
  // Indicate that reduction is done and D2H copy is done as well.
  bool local_used_maps_reduced_;

  // Work handle for allreduce on local_used_maps_
  c10::intrusive_ptr<c10d::ProcessGroup::Work> local_used_work_;

  void verify_replicas_within_process();

  void verify_replica0_across_processes();

  void mark_variable_ready_dense(VariableIndex index);

  void mark_variable_ready_sparse(VariableIndex index);

  void mark_variable_ready(VariableIndex index);

  void autograd_hook(VariableIndex index);

  void autograd_hook_compProcess(VariableIndex index);

  void mark_bucket_ready(size_t bucket_index);
  // compProcess
  void mark_bucket_ready_once(size_t bucket_index);

  void mark_bucket_ready_miniBatch(size_t bucket_index);
  void mark_bucket_ready_microBatch(size_t bucket_index);

  void finalize_bucket_dense(Bucket& replica);

  void finalize_backward();

  // Asserts that the reduction for the previous iteration has finished before
  // rebuilding buckets or kicking off the next one.
  void ensure_prior_reduction_finished();

  // Broadcast rebuilt buckets from rank 0 to other ranks before initializing
  // the buckets
  void sync_bucket_indices(std::vector<std::vector<size_t>>& bucket_indices);

  using GradCallback =
      torch::distributed::autograd::DistAutogradContext::GradCallback;
  void runGradCallbackForVariable(
      torch::autograd::Variable& variable,
      GradCallback&& cb);

  // A bucket replica represents [1..N] gradients to be reduced,
  // with the same dtype, on the same device.
  //
  // Batching gradients together before reducing them can result in lower
  // overhead and/or faster time to completion. Only gradients of the same type
  // and on the same device can be batched. The tensor that represents the
  // flattened gradient uses the same type and is placed on the same device.
  // Buckets are filled as the gradients they hold are computed (triggered by
  // autograd hooks). Buckets are reduced in a predetermined order that is
  // identical across processes.
  struct BucketReplica {
    // Flattened (1 dimensional) contents of bucket.
    at::Tensor contents;
    at::Tensor contents_cpu;

    // Views into contents for each grad.  Each view will be created with
    // layout (sizes + strides) matching the grad's expected layout
    // ("Gradient Layout Contract" in torch/csrc/autograd/AccumulateGrad.h).
    // `bucket_views_in[i].copy_(grad)` and
    // `grad.copy_(bucket_views_out[i])`
    // provide convenient ways to move grad data in/out of contents.
    // The reason we keep to states for bucket_views is that if DDP
    // communication hook was registered, `bucket_views_out` could be
    // re-initialized with the value of hook's `future_work`. We still need to
    // keep a separate view reference to replica's original contents for
    // `bucket_views_in[i].copy_(grad)` call.
    std::vector<at::Tensor> bucket_views_in;
    std::vector<at::Tensor> bucket_views_out;

    // Variables that contribute to this bucket replica. Use refcounted value
    // here so that we can easily unflatten the bucket contents into the
    // participating variables after reduction has completed.
    std::vector<torch::autograd::Variable> variables;

    // Per-variable offset/length into the flat bucket contents tensor and grad bucket.
    std::vector<size_t> offsets;
    std::vector<size_t> lengths;

    // Per-variable sizes into the grad bucekt.
    std::vector<c10::IntArrayRef> sizes_vec;

    // Number of tensors to be added before this bucket is complete.
    // This is reset to `variables.size()` every iteration.
    size_t pending;

    // TODO(@pietern)
    // Memory copies from gradient tensors into the bucket are potentially
    // done on different CUDA streams. We record an event for every copy
    // so that we can synchronize with them prior to kicking off the reduction.
    // std::vector<at::cuda::CUDAEvent> events;
  };

  // This function is called inside `initialize_buckets`, it initializes both
  // bucket_views_in and bucket_views_out into the contents tensor for each
  // variable's grad. Views serve as entry points to copy_ each grad's data
  // in/out of the flat contents tensor.
  void initialize_bucket_views(BucketReplica& replica, at::Tensor& contents);

  // This function is called inside `finalize_backward`, it happens only if
  // DDP communication hook was registered to recreate just bucket_views_out
  // with the result of `future_work`.
  void populate_bucket_views_out(BucketReplica& replica, at::Tensor& tensor);

  // If gradient_as_bucket_view_ is false, after allreduce buckets,
  // copy bucket results back to grads.
  void copy_bucket_to_grad(
      torch::autograd::Variable& variable,
      ElasticReducer::BucketReplica& replica,
      size_t intra_bucket_index,
      bool global_unused);
  void copy_bucket_to_grad_total_commProcess();
  // Check layout of grad and bucket_view before calling copy_grad_to_bucket
  void check_grad_layout(const at::Tensor& grad, const at::Tensor& bucket_view);
  // If gradient_as_bucket_view_ is false, before allreduce buckets,
  // copy grads to buckets.
  void copy_grad_to_bucket(const at::Tensor& grad, at::Tensor& bucket_view);
  void copy_grad_to_bucket_total_commProcess();

  // A bucket holds N bucket replicas (1 per model replica).
  //
  // If every bucket in this struct is ready, the reduction can be kicked off.
  // One bucket per replica. Reduction is kicked off when every bucket is ready.
  //
  struct Bucket {
    std::vector<BucketReplica> replicas;

    // Global indices of participating variables in the bucket
    std::vector<size_t> variable_indices;

    // Number of replicas to be marked done before this bucket is ready.
    size_t pending;

    // Keep work handle around when this set of buckets is being reduced.
    c10::intrusive_ptr<c10d::ProcessGroup::Work> work;

    // Keep future work handle around if DDP comm hook is registered.
    c10::intrusive_ptr<torch::jit::Future> future_work;

    // If this bucket should expect a single sparse gradient.
    // Implies: replicas[i].variables.size() == 1.
    bool expect_sparse_gradient = false;
  };

  std::vector<Bucket> buckets_;

  // A variable locator locates a particular variable in the bucket
  // structure. The `bucket_index` field points to the bucket in the `buckets_`
  // vector. The `intra_bucket_index` field points to the index of the variable
  // in any of the vector fields in the bucket replica.
  struct VariableLocator {
    // Index into the `buckets_` variable.
    size_t bucket_index;
    // Index of parameter in single bucket replica.
    size_t intra_bucket_index;

    VariableLocator() = default;

    VariableLocator(size_t bucket_index_, size_t intra_bucket_index_) {
      bucket_index = bucket_index_;
      intra_bucket_index = intra_bucket_index_;
    }
  };

  // Map the index of a variable to its location in the bucket structure.
  std::vector<VariableLocator> variable_locators_;

  // We collect the relative timestamp of every gradient being ready
  // when executing autograd. This can be used to derive a timeline of
  // the point in time buckets were ready, or ideal bucket assignment/ordering.
  int64_t backward_stats_base_;
  std::vector<std::vector<int64_t>> backward_stats_;

  // Following variables are to help build dynamic bucket order
  bool has_rebuilt_bucket_;
  std::vector<at::Tensor> rebuilt_params_;
  std::vector<int64_t> rebuilt_param_indices_;
  // The backup of rebuild_param_indices_, which is saved for checkpointing.
  std::vector<int64_t> rebuilt_param_indices_backup_;
  const int64_t bucket_bytes_cap_;

  struct RpcContext {
    using ContextPtr = torch::distributed::autograd::ContextPtr;
    // The shared_ptr is to hold the context instance.
    ContextPtr context_ptr_holder;
    std::atomic<ContextPtr::element_type*> context_ptr{nullptr};

    void set(ContextPtr&& new_context_ptr);
  };
  RpcContext rpc_context_;

  // A struct containing work handle and tensor for allreduce scheduled in
  // forward pass, if applicable.
  struct ForwardPassAllreduceWork {
    c10::intrusive_ptr<c10d::ProcessGroup::Work> workHandle;
    at::Tensor resultTensor;
    // whether we should divide by the initial world_size or the no. of
    // remaining DDP ranks.
    bool useStaticWorldSize;
  };

  // Handle for the currently scheduled allreduce in the forward pass, if
  // applicable.
  ForwardPassAllreduceWork forwardPassWorkHandle_;

  // Division factor for reduction of gradients.
  int divFactor_;
  
  // Whether this process is responsible for computation
  // 1 for computation process
  // 0 for communication process
  const bool is_computation_process_;

  // Process rank inside a GA.
  const int64_t local_rank_;
  const int64_t cheif_rank_;

  // Count the number of calling forward(mini_batch)
  int64_t cntMicroBatch_;

  // Logical gpu number (how many GA processes?)
  const int64_t logic_gpus_num_;

  // Shm for signal communication in a single pod
  at::Tensor shm_array_;

  // Shm of grads in a single pod
  std::vector<at::Tensor> shm_grads_;

  mutable std::mutex order_mutex_;

  bool is_grad_assigned_ = false;

  bool is_bucketOrder_dumped_ = false;

  int64_t cnt_bucketOrder_ = 0;

  std::vector<int64_t> bucket_allreduce_order_;

  bool should_force_rebuild_buckets_ = false;

  bool is_rebuilt_param_indices_saved_ = false;

  bool is_force_rebuild_buckets_finished_ = false;

  void append_allreduce_order(const int64_t bucket_index);

  bool wait_for_shm_signal(const int signal, const int pos) const;

  void customized_broadcast(std::vector<int64_t>& data_vec, const int src_rank);

  void do_allreduce_commProcess();


 private:
  // comm_hook_ is used to access the DDP communication hook if registered.
  std::unique_ptr<CommHookInterface> comm_hook_;

  // ddp_logging_data_ is used to hold all the ddp related logging
  // data fields.
  std::unique_ptr<c10::DDPLoggingData> ddp_logging_data_;
};


} // namespace c10d
