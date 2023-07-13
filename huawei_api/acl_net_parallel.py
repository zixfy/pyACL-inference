# Any bug can be reported at http://gitlab.buaadml.info/bdi/pyacl-inference/-/issues
# Doc: http://gitlab.buaadml.info/bdi/pyacl-inference/
import sys
from typing import List, Optional, Any, Dict, Tuple, Union, NoReturn, Type, Callable
import numpy as np
import acl
import threading

# error code
ACL_SUCCESS = 0

# rule for mem
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEM_MALLOC_HUGE_ONLY = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 2

# rule for memory copy
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3

ACL_NONBLOCK = 0
ACL_BLOCK = 1

NPU_IO_DEBUG = False
NPU_RESULT_DEBUG = False
output_result_mutex = threading.Lock()
buffer_method = {
    "in": acl.mdl.get_input_size_by_index,
    "out": acl.mdl.get_output_size_by_index
}

acl_dtype_idx_to_name = {
    0: 'ACL_FLOAT',
    1: 'ACL_FLOAT16',
    2: 'ACL_INT8',
    3: 'ACL_INT32',
    4: 'ACL_UINT8',
    6: 'ACL_INT16',
    7: 'ACL_UINT16',
    8: 'ACL_UINT32',
    9: 'ACL_INT64',
    10: 'ACL_UINT64',
    11: 'ACL_DOUBLE',
    12: 'ACL_BOOL'}

acl_dtype_idx_to_numpy_dtype = {
    0: np.single,
    1: np.float16,
    2: np.int8,
    3: np.int32,
    4: np.uint8,
    6: np.int16,
    7: np.uint16,
    8: np.uint32,
    9: np.int64,
    10: np.uint64,
    11: np.double,
    12: np.bool_
}

# https://support.huawei.com/enterprise/zh/doc/EDOC1100164876/1986f59c
acl_dtype_idx_to_numpy_dtype_idx = {
    0: 11,
    1: 23,
    2: 1,
    3: 5,
    4: 2,
    6: 3,
    7: 4,
    8: 6,
    9: 7,
    10: 8,
    11: 12,
    12: 0
}


def check_ret(message: str, ret: int):
    """
    Huawei official implement,
    check the return value of a call of pyACL API
    """
    if ret != ACL_SUCCESS:
        raise Exception("{} failed ret={}"
                        .format(message, ret))


def to_aligned_str(s: str, aligned_len: int = 50, padding_len: int = 15, padding_char_left: str = ">",
                   padding_char_right='<') -> str:
    """ A helper to format center-aligned string

    :param s: input
    :param aligned_len: s is first center-padded to at least {aligned_len} chars
    :param padding_len: s is then padded with {padding_len} numbers of {padding_char_left/right} in both sides
    :param padding_char_left/right: (Above)
    :returns: Formatted string of s
    """
    s = s.center(max(len(s), aligned_len), " ")
    return padding_char_left * padding_len + s + padding_char_right * padding_len


def require_npu_context(func: Callable) -> Callable:
    """A decorator for public methods in 'Net' class
    Because Any call of pyacl API must set correct npu context for different model id and device id
    :param func: public method in 'Net' class that invokes any pyacl API function
    :return: a closure of decorated function that set context correctly ahead
    """

    def wrapper(self, *args, **kwargs):
        acl.rt.set_context(self.context)
        return func(self, *args, **kwargs)

    return wrapper


def using_huawei_api(func: Callable) -> Callable:
    """A decorator for any function that uses 'ACLNetHandler' class Since acl.init() / acl.finalize must be called
    once in every process, so decorate any main runner function in process that use huawei model
    :param func: runner function inferring to huawei model/API, like FeatureExtractor.run in project
    :return: a closure of decorated function that setup and clear huawei pyACL things correctly
    """

    def wrapper(*args, **kwargs):
        init_huawei_api()
        res = func(*args, **kwargs)
        finalize_huawei_api()
        return res

    return wrapper


class Net(object):
    """
    Acknowledgement: This class is a modification of original Huawei official implication:
    https://gitee.com/ascend/samples/blob/master/python/level2_simple_inference/1_classification/resnet50_async_imagenet_classification/src/acl_net.py
    The wrapper of pyACL runtime API, private in this package

    Note: never care this class, use class ACLNetHandler below for wrapping model inference
    """

    def __init__(self, device_id: int, model_path: str):
        """Specify the NPU and .om model path, load model
        :param device_id: logical number of NPU, which can be obtained with command 'npu-smi info'
        :param model_path: any .om model path, only single input models are supported now
        """

        # setter
        # dataset_list is to hold C-pointer to input/output's buffer memory on NPU
        self.dataset_list: List[Tuple[int, int]] = []
        self.device_id: int = device_id
        self.model_path: str = model_path

        # Explicitly setup the Device(NPU) used for inference
        print(to_aligned_str("[NPU:{}] init resource stage".format(self.device_id)))
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        # context is wrapper of Device, Stream, Event
        # (https://support.huawei.com/enterprise/zh/doc/EDOC1100164876/98fecc8)
        # Note: context is needed to be switched with acl.rt.set_context when switching different NPU-Device / Model
        # Note: context is unique in every thread
        self.context: int
        self.context, ret = acl.rt.create_context(self.device_id)
        ret1 = check_ret("acl.rt.create_context", ret)

        # setup Stream, which guarantees the synchronization of parallel computation in pyACL
        self.stream: int
        self.stream, ret = acl.rt.create_stream()
        check_ret("acl.rt.create_stream", ret)

        # load model from file to NPU, and get a unique ID for held model
        self.model_id: int
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)

        # get a aclmdlDesc* pointer in underlying C-implement
        # aclmdlDesc is used to read IO information about model
        # Note: it's a c-pointer only for pyACL API
        self.model_desc: int
        self.model_desc = acl.mdl.create_desc()

        # initial aclmdlDesc with model information
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        check_ret("acl.mdl.get_desc", ret)

        # print format of model's input/output for logging
        self.input_num: int
        self.output_num: int
        self.input_acl_dtype_idx: int
        self.input_numpy_dtype: Type
        self.output_acl_dtype_idxs: List[int]
        self.output_numpy_dtypes: List[Type]
        self.output_dims: List[Tuple]
        info = self._print_model_info()
        print(to_aligned_str("[NPU:{}] init resource success".format(self.device_id)))
        print()

    def __del__(self):
        """
        release CPU/NPU memory
        """
        print(to_aligned_str("[NPU:{}] Releasing NPU resources stage".format(self.device_id)))
        # always explicitly del Net object to release NPU in-time, and avoid this branch
        if acl.rt is None:
            print(to_aligned_str('[NPU:{}] Resources released By Python Interpreter.'.format(self.device_id)))
            return

        # Note: always acl.rt.set_context firstly since we are using multi-NPUs for parallel
        acl.rt.set_context(self.context)

        # kill model
        if self.model_id:
            ret = acl.mdl.unload(self.model_id)
            check_ret("acl.mdl.unload", ret)

        if self.model_desc:
            ret = acl.mdl.destroy_desc(self.model_desc)
            check_ret("acl.mdl.destroy_desc", ret)

        # kill descriptor in pyACL
        if self.stream:
            ret = acl.rt.destroy_stream(self.stream)
            check_ret("acl.rt.destroy_stream", ret)
        if self.context:
            ret = acl.rt.destroy_context(self.context)
            check_ret("acl.rt.destroy_context", ret)
        ret = acl.rt.reset_device(self.device_id)
        check_ret("acl.rt.reset_device", ret)

        print(to_aligned_str('[NPU:{}] Resources released successfully.'.format(self.device_id)))
        print()

    def _print_model_info(self) -> NoReturn:
        """print model's information
        The number of input/output, the shape(dims) and data type of each input/output are logged
        Note: only models with single input are supported now
        Note: check if it's same with .onnx, .pt model
        """
        self.input_num: int = acl.mdl.get_num_inputs(self.model_desc)
        self.output_num: int = acl.mdl.get_num_outputs(self.model_desc)
        self.input_acl_dtype_idx = acl.mdl.get_input_data_type(self.model_desc, 0)
        self.input_numpy_dtype = acl_dtype_idx_to_numpy_dtype[self.input_acl_dtype_idx]
        self.output_acl_dtype_idxs = [acl.mdl.get_output_data_type(self.model_desc, i) for i in range(self.output_num)]
        self.output_numpy_dtypes = [acl_dtype_idx_to_numpy_dtype[i] for i in self.output_acl_dtype_idxs]
        self.output_dims = [tuple(acl.mdl.get_output_dims(self.model_desc, i)[0]["dims"]) for i in
                            range(self.output_num)]
        print("• Model path: ", self.model_path)
        print("• Input total: ", self.input_num)
        for i in range(self.input_num):
            print("└─── input {}: shape={}, dtype={}".format(i, acl.mdl.get_input_dims(self.model_desc, i)[0],
                                                             acl_dtype_idx_to_name[
                                                                 acl.mdl.get_input_data_type(self.model_desc, i)]))
        print("• Output total: ", self.output_num)
        for i in range(self.output_num):
            print("└─── output {}: shape={}, dtype={}".format(i, acl.mdl.get_output_dims(self.model_desc, i)[0],
                                                              acl_dtype_idx_to_name[self.output_acl_dtype_idxs[i]]))

    def _load_input_data(self, images_batch: np.ndarray) -> int:
        """Copy numpy input to NPU memory
        :param images_batch:List of input image batch, every batch's batch size must match .om model's batch size
        Note: only single input flow are supported now.
        :return: a pointer to aclmdlDataset, which holds buffer of inference input on NPU memory
        """

        # convert numpy.ndarray into correct data type

        # print("flags['OWNDATA'] before copy： {}", images_data.flags['OWNDATA'])
        if images_batch.dtype != self.input_numpy_dtype:
            images_batch = images_batch.astype(self.input_numpy_dtype)

        # Note: it's a undefined behavior to pass a numpy array view to acl.util.numpy_to_ptr,
        #   so make sure numpy array OWN DATA
        if not images_batch.flags['OWNDATA']:
            images_batch = images_batch.copy()
        # print("flags['OWNDATA'] after copy： {}", images_data.flags['OWNDATA'])

        # get C-pointer to ndarray data in numpy implement
        if "bytes_to_ptr" in dir(acl.util):
            bytes_data = images_batch.tobytes()
            img_ptr = acl.util.bytes_to_ptr(bytes_data)
        else:
            img_ptr = acl.util.numpy_to_ptr(images_batch)

        # memcpy image ndarray to malloced memory on NPU
        image_buffer_size = images_batch.size * images_batch.itemsize
        img_device, ret = acl.rt.malloc(image_buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY)
        check_ret("acl.rt.malloc", ret)
        ret = acl.rt.memcpy(img_device, image_buffer_size, img_ptr,
                            image_buffer_size, ACL_MEMCPY_HOST_TO_DEVICE)
        check_ret("acl.rt.memcpy", ret)

        # create aclmdlDataset C-pointer
        img_dataset = acl.mdl.create_dataset()
        img_data_buffer = acl.create_data_buffer(img_device, image_buffer_size)
        _, ret = acl.mdl.add_dataset_buffer(img_dataset, img_data_buffer)
        if ret != ACL_SUCCESS:
            ret = acl.destroy_data_buffer(img_data_buffer)
            check_ret("acl.destroy_data_buffer", ret)

        return img_dataset

    def _load_output_data(self) -> int:
        """Malloc memory on NPU to store inference output, indeterminate memory during inference need not be cared
        :return: a pointer to aclmdlDataset, which holds buffer of inference output on NPU memory
        """
        output_data = acl.mdl.create_dataset()

        # support multi-outputs model
        for i in range(self.output_num):
            temp_buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY)
            check_ret("acl.rt.malloc", ret)

            data_buf = acl.create_data_buffer(temp_buffer, temp_buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(output_data, data_buf)
            if ret != ACL_SUCCESS:
                ret = acl.destroy_data_buffer(data_buf)
                check_ret("acl.destroy_data_buffer", ret)
        return output_data

    def _load_data_to_npu(self, images_batches: List[np.ndarray]) -> List[Tuple[int, int]]:
        """ load numpy.ndarray on CPU memory to NPU memory
        :param images_batches: List of input image batch, every batch's batch size must match .om model's batch size
        :return: List of input/output buffer pointer on NPU for each batch
        """
        dataset_list: List[Tuple[int, int]] = []
        for batch in images_batches:
            input_dataset_ptr = self._load_input_data(batch)
            output_dataset_ptr = self._load_output_data()
            dataset_list.append((input_dataset_ptr, output_dataset_ptr))
        return dataset_list

    @require_npu_context
    def dispatch_parallel_job(self, image_batches: List[np.ndarray]) -> NoReturn:
        """Dispatch a list of batches for inference on self.context
        :param image_batches: List of input image batch, every batch's batch size must match .om model's batch size
        """
        # copy images to device
        self.dataset_list = self._load_data_to_npu(image_batches)

        # async parallel computation
        for input_dataset_ptr, output_dataset_ptr in self.dataset_list:
            ret = acl.mdl.execute_async(self.model_id, input_dataset_ptr, output_dataset_ptr, self.stream)
            check_ret("acl.mdl.execute_async", ret)

    @staticmethod
    def _destroy_dataset(dataset_ptr: int) -> NoReturn:
        """Free NPU memory using aclmdlDataset Pointer
         :param dataset_ptr:  ptr to input / output of inference
        """
        num_of_data_buf = acl.mdl.get_dataset_num_buffers(dataset_ptr)
        for i in range(num_of_data_buf):
            data_buf = acl.mdl.get_dataset_buffer(dataset_ptr, i)
            if data_buf:
                data = acl.get_data_buffer_addr(data_buf)
                ret = acl.rt.free(data)
                check_ret("acl.rt.free", ret)
                ret = acl.destroy_data_buffer(data_buf)
                check_ret("acl.destroy_data_buffer", ret)
        ret = acl.mdl.destroy_dataset(dataset_ptr)
        check_ret("acl.mdl.destroy_dataset", ret)

    @require_npu_context
    def fetch_output_from_npu(self) -> List[List[np.ndarray]]:
        """Synchronize inference computation on each parallelled NPU device, and get inference output
        :return: 2-dim List of each batch's numpy single or multi   (on CPU)
        Note: single inference output is also a list of one ndarray
        """

        # Synchronize computation
        ret = acl.rt.synchronize_stream(self.stream)
        check_ret("acl.rt.synchronize_stream", ret)
        inference_outputs: List[List[np.ndarray]] = []

        # for each batch
        for dataset in self.dataset_list:
            input_dataset_ptr, output_dataset_ptr = dataset
            current_inference_output: List[np.ndarray] = []

            # memcpy inference output from NPU to CPU
            num_of_data_buf = acl.mdl.get_dataset_num_buffers(output_dataset_ptr)
            for i in range(num_of_data_buf):
                temp_output_buf = acl.mdl.get_dataset_buffer(output_dataset_ptr, i)
                infer_output_ptr = acl.get_data_buffer_addr(temp_output_buf)
                infer_output_size = acl.get_data_buffer_size_v2(temp_output_buf)

                # malloc on CPU memory
                ptr, ret = acl.rt.malloc_host(infer_output_size)
                check_ret("acl.rt.malloc_host", ret)
                ret = acl.rt.memcpy(ptr,
                                    infer_output_size,
                                    infer_output_ptr,
                                    infer_output_size,
                                    ACL_MEMCPY_DEVICE_TO_HOST)
                check_ret("acl.rt.memcpy", ret)
                dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, i)
                check_ret("acl.mdl.get_cur_output_dims", ret)

                # setup correct data type and shape for each output flow in model
                if "ptr_to_bytes" in dir(acl.util):
                    bytes_data = acl.util.ptr_to_bytes(ptr, infer_output_size)
                    data = np.frombuffer(bytes_data, dtype=self.output_numpy_dtypes[i]).reshape(self.output_dims[i])
                else:
                    data = acl.util.ptr_to_numpy(ptr, self.output_dims[i],
                                                 acl_dtype_idx_to_numpy_dtype_idx[self.output_acl_dtype_idxs[i]])

                # free acl.rt.malloc_host(), to avoid memory leakage
                current_inference_output.append(data.copy())
                acl.rt.free_host(ptr)
            # free memory buffer on NPU
            self._destroy_dataset(input_dataset_ptr)
            self._destroy_dataset(output_dataset_ptr)
            inference_outputs.append(current_inference_output)
        return inference_outputs

    @require_npu_context
    def synchronize_device(self) -> NoReturn:
        # synchronize NPU devices (according to official guide)
        acl.rt.synchronize_device(self.device_id)

    @require_npu_context
    def get_input_batch_size(self) -> int:
        """Just get batch size for computation jobs dispatching
        :return:batch size of .om model
        """
        return acl.mdl.get_input_dims(self.model_desc, 0)[0]["dims"][0]


# Exposed to outer of package
class ACLNetHandler(object):
    """Model loading, Memory Manage and Parallel Inference for .om model
    1. Use ACLNetHandler to specify the NPU logic number used in inference, which can be found
    using 'npu-smi info' in CANN
    2. Due to the use constraints of pyACL API, each process using the Huawei model must invoke
    init_huawei_api/finalize_huawei_api at the beginning/end, or use using_huawei_api to decorate the main runner
    function of the process
    3. Inference input Type is np.ndarray of shape [batch, ...]; calling forward() or __call__() will try to use the
    specified npu_devices for parallel inference; The inference output of the single-output model is
    np.ndarray, and the output of the multi-output model is List[np.ndarray]. For specific data-type and shape,
    please refer to the information printed when loading the model
    4. ACLNetHandler manages NPU resources, whose destruction must rely on acl.rt / acl.mdl, so you need to explicitly
    use release() or __del__() method to unload model instead of relying on python-GCe/type,
    """

    def __init__(self, npu_device_ids: Union[List[int], int], om_model_path: str):
        """Load model from file, specify NPU device for inference
        :param npu_device_ids: logic NPU number(s), not physical NPU number(s), typically: [0,1] or 0
        :param om_model_path: model to be Data&Model paralleled
        """
        # print(om_model_path)
        self.npu_device_ids = [npu_device_ids] if isinstance(npu_device_ids, int) else npu_device_ids
        if len(self.npu_device_ids) == 0:
            raise RuntimeError('no NPU specified')
        if len(self.npu_device_ids) > len(set(self.npu_device_ids)):
            raise RuntimeError('duplicated NPU device IDs')
        self.acl_net = [Net(i, om_model_path) for i in self.npu_device_ids]
        self.batch_size = self.acl_net[0].get_input_batch_size()

    def forward(self, images_data: np.ndarray) -> Union[List[np.ndarray], np.ndarray]:
        """Parallel inference

        :param images_data: np.ndarray with shape [batch, *dims], axis-0 (batch) is dispatched to at most len(
        self.acl_net) for parallel computation
        Note: batch that can't be divided by .om model's batch_size is also okay
        :return:The inference output of the single-output model is np.ndarray,
        and the output of the multi-output model is List[np.ndarray]
        """

        # padding to fir .om model's batch_size
        real_batch_size: int = images_data.shape[0]
        rem: int = real_batch_size % self.batch_size
        if rem != 0:
            padding_config = tuple(
                (0, self.batch_size - rem) if i == 0 else (0, 0) for i in range(len(images_data.shape)))
            # Pad zeros along the 0th axis
            images_data = np.pad(images_data, padding_config, mode='constant')

        # split input many batches
        jobs = [images_data[i:i + self.batch_size] for i in range(0, images_data.shape[0], self.batch_size)]
        jobs_num = len(jobs)
        real_npu_num = min(len(self.acl_net), jobs_num)
        q: int = jobs_num // real_npu_num
        r: int = jobs_num % real_npu_num

        # dispatch batches fairly to {real_npu_num} NPUs
        dispatched_jobs: List[List[np.ndarray]] = []
        idx = 0
        for i in range(real_npu_num):
            nxt: int = idx + q + (1 if i < r else 0)
            dispatched_jobs.append(jobs[idx:nxt])
            idx = nxt

        # start inference on each NPU
        result = self._forward(dispatched_jobs)

        # assume input / output has same batch_size
        for i in range(len(result)):
            result[i] = result[i][:real_batch_size, ...]
        if len(result) == 1:
            result = result[0]
        return result

    def _forward(self, images_batches_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Invoke execute_async on NPU, and synchronize NPU devices to get results

        :param images_batches_list: List of dispatched batches for each used NPU
        :return:List of inference outputs of model
        """
        real_npu_num = len(images_batches_list)

        # synchronize, and gather output
        for i in range(real_npu_num):
            self.acl_net[i].dispatch_parallel_job(images_batches_list[i])
        tmp = self.acl_net[0].fetch_output_from_npu()
        output: List[List[np.ndarray]] = [list(row) for row in zip(*tmp)]
        for i in range(1, real_npu_num):
            for batch_outputs in self.acl_net[i].fetch_output_from_npu():
                for output_idx, batch_output in enumerate(batch_outputs):
                    output[output_idx].append(batch_output)

        # synchronize device API
        for i in range(real_npu_num):
            self.acl_net[i].synchronize_device()

        # cat gathered result
        stacked_output = [np.concatenate(o, axis=0) for o in output]
        return stacked_output

    def __call__(self, batch_images: Union[List[np.ndarray], np.ndarray]) \
            -> Union[List[np.ndarray], np.ndarray]:
        # proxy
        return self.forward(batch_images)

    def __del__(self):
        # proxy
        self.release()

    def release(self):
        while len(self.acl_net) > 0:
            del self.acl_net[-1]


# Exposed to outer of package
def init_huawei_api() -> NoReturn:
    # Note: must be manually invoked in every process's beginning
    ret = acl.init()
    check_ret("acl.init", ret)


# Exposed to outer of package
def finalize_huawei_api() -> NoReturn:
    # Exposed to outer of package
    # Note: must be manually invoked in every process's ending
    ret = acl.finalize()
    check_ret("acl.finalize", ret)
