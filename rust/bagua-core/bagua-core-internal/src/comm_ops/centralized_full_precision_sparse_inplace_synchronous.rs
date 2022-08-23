use crate::comm_ops::CommOpTrait;
use crate::communicators::BaguaCommunicator;
use crate::datatypes::{BaguaBucket, BaguaReductionOp, BaguaTensor, BaguaTensorRaw, RawBaguaTensor};
use crate::resource_pool::CUDA_DEVICE_MEMORY_POOL;
use crate::BaguaCommOpChannels;
use crate::kernels;
use std::sync::Arc;

#[derive(Debug)]
pub struct CentralizedFullPrecisionSparseInplaceSynchronous {
    pub communicator: BaguaCommunicator,
    pub other_tensor: BaguaTensor,
}

impl CommOpTrait for CentralizedFullPrecisionSparseInplaceSynchronous {
    fn execute_background_communication(
        &self,
        bucket: Arc<BaguaBucket>,
        _comm_op_channels: &BaguaCommOpChannels,
    ) {
        let bucket = bucket.inner.lock();
        let stream_ptr = self.communicator.stream_ptr();
        let mut communication_tensor = bucket.get_communication_tensor(stream_ptr, false, false);
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
                println!("index tensor raw length: {} before.", t.raw.num_elements());
                let temp_other_buf = CUDA_DEVICE_MEMORY_POOL[other_tensor_raw.device_id()]
                    .try_pull(t.raw.num_elements_allocated() * other_tensor_raw.dtype().bytes())
                    .expect("cannot allocate cuda memory");

                unsafe {
                    kernels::sparse_inplace_extract_host(
                        other_tensor_raw.data_ptr() as _,
                        t.raw.data_ptr() as _,
                        t.raw.num_elements() as _,
                        temp_other_buf.ptr as _,
                        c.stream_ptr as _,
                    );
                }

                let mut send_other_tensor = BaguaTensorRaw {
                    ptr: temp_other_buf.ptr,
                    num_elem_allocated: t.raw.num_elements_allocated(),
                    dtype: other_tensor_raw.dtype().clone(),
                    num_elem: t.raw.num_elements(),
                    device_id: t.raw.device_id(),
                    pool_allocations: vec![Arc::new(temp_other_buf)],
                };

                let recv_other_buf = CUDA_DEVICE_MEMORY_POOL[other_tensor_raw.device_id()]
                    .try_pull(send_other_tensor.num_elements_allocated() * send_other_tensor.dtype().bytes() * c.nranks)
                    .expect("cannot allocate cuda memory");

                let mut recv_others_tensor = BaguaTensorRaw {
                    ptr: recv_other_buf.ptr,
                    num_elem_allocated: send_other_tensor.num_elements_allocated() * c.nranks,
                    dtype: other_tensor_raw.dtype().clone(),
                    num_elem: send_other_tensor.num_elements() * c.nranks,
                    device_id: send_other_tensor.device_id(),
                    pool_allocations: vec![Arc::new(recv_other_buf)],
                };

                c.allgather(&mut send_other_tensor, &mut recv_others_tensor);

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
                    kernels::sparse_inplace_gather_host(
                        recv_others_tensor.data_ptr() as _,
                        recv_index_tensor.data_ptr() as _,
                        recv_index_tensor.num_elements() as _,
                        other_tensor_raw.data_ptr() as _,
                        other_tensor_raw.num_elements() as _,
                        c.stream_ptr as _,
                    );
                }

                tracing::debug!("internode communication done")
            },
        );
    }
}
