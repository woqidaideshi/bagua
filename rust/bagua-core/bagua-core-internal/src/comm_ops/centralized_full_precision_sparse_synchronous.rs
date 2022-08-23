use crate::comm_ops::CommOpTrait;
use crate::communicators::BaguaCommunicator;
use crate::datatypes::{BaguaBucket, BaguaReductionOp, BaguaTensor, BaguaTensorRaw, RawBaguaTensor};
use crate::resource_pool::CUDA_DEVICE_MEMORY_POOL;
use crate::BaguaCommOpChannels;
use crate::kernels;
use std::sync::Arc;

#[derive(Debug)]
pub struct CentralizedFullPrecisionSparseSynchronous {
    pub communicator: BaguaCommunicator,
    pub value_tensor: BaguaTensor,
    pub other_tensor: BaguaTensor,
}

impl CommOpTrait for CentralizedFullPrecisionSparseSynchronous {
    fn execute_background_communication(
        &self,
        bucket: Arc<BaguaBucket>,
        _comm_op_channels: &BaguaCommOpChannels,
    ) {
        let bucket = bucket.inner.lock();
        let stream_ptr = self.communicator.stream_ptr();
        let mut communication_tensor = bucket.get_communication_tensor(stream_ptr, false, false);
        let mut value_guard = self.value_tensor.inner.write();
        let mut value_tensor_raw = value_guard.raw.as_mut();
        let mut other_guard = self.other_tensor.inner.write();
        let mut other_tensor_raw = other_guard.raw.as_mut();
        self.communicator.execute_communication(
            &mut communication_tensor,
            false,
            true,
            true,
            &mut |c, t| {
                tracing::debug!("internode communication started");
                tracing::debug!("start alltoall");

                let recv_value_buf = CUDA_DEVICE_MEMORY_POOL[value_tensor_raw.device_id()]
                    .try_pull(value_tensor_raw.num_elements_allocated() * value_tensor_raw.dtype().bytes() * c.nranks)
                    .expect("cannot allocate cuda memory");

                let mut recv_value_tensor = BaguaTensorRaw {
                    ptr: recv_value_buf.ptr,
                    num_elem_allocated: value_tensor_raw.num_elements_allocated() * c.nranks,
                    dtype: value_tensor_raw.dtype().clone(),
                    num_elem: value_tensor_raw.num_elements() * c.nranks,
                    device_id: value_tensor_raw.device_id(),
                    pool_allocations: vec![Arc::new(recv_value_buf)],
                };

                c.allgather(value_tensor_raw, &mut recv_value_tensor);

                let recv_index_buf = CUDA_DEVICE_MEMORY_POOL[t.raw.device_id()]
                    .try_pull(t.raw.num_elements_allocated() * t.raw.dtype().bytes() * c.nranks)
                    .expect("cannot allocate cuda memory");

                let mut recv_index_tensor = BaguaTensorRaw {
                    ptr: recv_index_buf.ptr,
                    num_elem_allocated: t.raw.num_elements_allocated() * c.nranks,
                    dtype: t.raw.dtype().clone(),
                    num_elem: t.raw.num_elements() * c.nranks,
                    device_id: t.raw.device_id(),
                    pool_allocations: vec![Arc::new(recv_index_buf)],
                };
                c.allgather(&mut t.raw, &mut recv_index_tensor);

                unsafe {
                    kernels::sparse_gather_host(
                        recv_value_tensor.data_ptr() as _,
                        recv_index_tensor.data_ptr() as _,
                        recv_index_tensor.num_elements() as _,
                        other_tensor_raw.data_ptr() as _,
                        0 as _,
                        c.stream_ptr as _,
                    );
                }

                other_tensor_raw.divide_inplace(c.stream_ptr, c.nranks as f32);

                tracing::debug!("internode communication done")
            },
        );
    }
}
