pub mod centralized_full_precision_synchronous;
pub mod centralized_low_precision_synchronous;
pub mod decentralized_full_precision_asynchronous;
pub mod decentralized_full_precision_synchronous;
pub mod decentralized_low_precision_synchronous;
pub mod centralized_middle_precision_synchronous;
pub mod centralized_full_precision_sparse_synchronous;
pub mod centralized_full_precision_sparse_inplace_synchronous;
pub mod centralized_full_precision_sparse_py_synchronous;
pub mod centralized_full_precision_sparse_py_cuda_synchronous;
pub mod centralized_full_precision_sparse_py_rust_synchronous;
pub mod centralized_full_precision_sparse_py_cuda_parallel_synchronous;
pub mod centralized_full_precision_sparse_inplace_parallel_synchronous;
pub mod python_ffi_op;

use crate::datatypes::BaguaBucket;
use crate::{BaguaCommOpChannels, BaguaCoreError};
use std::fmt::Debug;
use std::sync::Arc;

pub trait CommOpTrait: Debug {
    fn execute_background_communication(
        &self,
        bucket: Arc<BaguaBucket>,
        comm_channels: &BaguaCommOpChannels,
    );
}
