# Any bug can be reported at http://gitlab.buaadml.info/bdi/pyacl-inference/-/issues
# Doc: http://gitlab.buaadml.info/bdi/pyacl-inference/
import sys
from typing import List, Optional, Any, Dict, Tuple, Union, NoReturn, Type
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
    Huawei official implication,
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


def require_npu_context(func):
    def wrapper(self, *args, **kwargs):
        acl.rt.set_context(self.context)
        return func(self, *args, **kwargs)
    return wrapper


def using_huawei_api(func):
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

    def __init__(self, device_id: int,
                 model_path: str
                 ):
        """Specify the NPU and .om model path, load model
        :param device_id: logical number of NPU, which can be obtained with command 'npu-smi info'
        :param model_path: any .om model path, only single input models are supported now
        """

        # setter
        self.dataset_list: List[Tuple[int, int]] = []
        self.device_id: int = device_id
        self.model_path: str = model_path

        # Explicitly specifying the Device(NPU) used for inference
        print(to_aligned_str("[NPU:{}] init resource stage".format(self.device_id)))
        ret = acl.rt.set_device(self.device_id)
        check_ret("acl.rt.set_device", ret)

        # context is wrapper of Device, Stream, Event
        # (https://support.huawei.com/enterprise/zh/doc/EDOC1100164876/98fecc8)
        # Note: context is needed to be switched with acl.rt.set_context when switching different NPU-Device / Model
        # Note: context is unique in every thread
        self.context: int
        self.context, ret = acl.rt.create_context(self.device_id)
        check_ret("acl.rt.create_context", ret)

        # setup Stream, which guarantees the synchronization of parallel computation in pyACL
        self.stream: int
        self.stream, ret = acl.rt.create_stream()
        check_ret("acl.rt.create_stream", ret)

        # load model from file to NPU, and get a unique ID for held model
        self.model_id: int
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        check_ret("acl.mdl.load_from_file", ret)

        # get a aclmdlDesc* pointer in underlying C-implement
        # aclmdlDesc is used to read
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
        self._print_model_info()
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

    @require_npu_context
    def get_input_batch_size(self):
        return acl.mdl.get_input_dims(self.model_desc, 0)[0]["dims"][0]

    def _load_input_data(self, images_data):
        # images_data = images_data.astype(self.input_numpy_dtype)
        # print("flags['OWNDATA'] before copy： {}", images_data.flags['OWNDATA'])
        if images_data.dtype != self.input_numpy_dtype:
            images_data = images_data.astype(self.input_numpy_dtype)
        if not images_data.flags['OWNDATA']:
            images_data = images_data.copy()
        # print("flags['OWNDATA'] after copy： {}", images_data.flags['OWNDATA'])
        if "bytes_to_ptr" in dir(acl.util):
            bytes_data = images_data.tobytes()
            img_ptr = acl.util.bytes_to_ptr(bytes_data)
        else:
            img_ptr = acl.util.numpy_to_ptr(images_data)  # host ptr
        # memcopy host to device
        image_buffer_size = images_data.size * images_data.itemsize

        img_device, ret = acl.rt.malloc(image_buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY)
        check_ret("acl.rt.malloc", ret)
        ret = acl.rt.memcpy(img_device, image_buffer_size, img_ptr,
                            image_buffer_size, ACL_MEMCPY_HOST_TO_DEVICE)
        check_ret("acl.rt.memcpy", ret)

        # create dataset in device
        img_dataset = acl.mdl.create_dataset()
        img_data_buffer = acl.create_data_buffer(img_device, image_buffer_size)
        _, ret = acl.mdl.add_dataset_buffer(img_dataset, img_data_buffer)
        if ret != ACL_SUCCESS:
            ret = acl.destroy_data_buffer(img_data_buffer)
            check_ret("acl.destroy_data_buffer", ret)

        return img_dataset

    def _load_output_data(self):
        output_data = acl.mdl.create_dataset()
        for i in range(self.output_num):
            # check temp_buffer dtype
            temp_buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, i)

            # Note: malloc on NPU, not CPU
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY)
            check_ret("acl.rt.malloc", ret)

            data_buf = acl.create_data_buffer(temp_buffer, temp_buffer_size)
            _, ret = acl.mdl.add_dataset_buffer(output_data, data_buf)
            if ret != ACL_SUCCESS:
                ret = acl.destroy_data_buffer(data_buf)
                check_ret("acl.destroy_data_buffer", ret)
        return output_data

    def _load_data_to_npu(self, images_batches) -> NoReturn:
        if NPU_IO_DEBUG:
            print("data interaction from host to device")
        dataset_list = []
        for batch in images_batches:
            input_dataset_ptr = self._load_input_data(batch)
            output_dataset_ptr = self._load_output_data()
            dataset_list.append((input_dataset_ptr, output_dataset_ptr))
        if NPU_IO_DEBUG:
            print("data interaction from host to device success")
        return dataset_list

    @staticmethod
    def _destroy_dataset(dataset_ptr: int):
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
    def dispatch_parallel_job(self, image_batches: List[np.ndarray]):
        # copy images to device
        self.dataset_list = self._load_data_to_npu(image_batches)
        if NPU_IO_DEBUG:
            print('execute stage:')
        for input_dataset_ptr, output_dataset_ptr in self.dataset_list:
            # ret = acl.mdl.execute(self.model_id, input_dataset_ptr, output_dataset_ptr)
            # check_ret("acl.mdl.execute", ret)
            ret = acl.mdl.execute_async(self.model_id, input_dataset_ptr, output_dataset_ptr, self.stream)
            check_ret("acl.mdl.execute_async", ret)
        if NPU_IO_DEBUG:
            print('execute stage success')

    @require_npu_context
    def fetch_output_from_npu(self):
        ret = acl.rt.synchronize_stream(self.stream)
        check_ret("acl.rt.synchronize_stream", ret)

        if NPU_IO_DEBUG:
            print('callback func stage:')
        inference_outputs: List[List[np.ndarray]] = []
        for dataset in self.dataset_list:
            input_dataset_ptr, output_dataset_ptr = dataset
            current_inference_output: List[np.ndarray] = []
            # device to host
            num_of_data_buf = acl.mdl.get_dataset_num_buffers(output_dataset_ptr)
            for i in range(num_of_data_buf):
                temp_output_buf = acl.mdl.get_dataset_buffer(output_dataset_ptr, i)
                infer_output_ptr = acl.get_data_buffer_addr(temp_output_buf)
                infer_output_size = acl.get_data_buffer_size_v2(temp_output_buf)
                output_host, ret = acl.rt.malloc_host(infer_output_size)
                check_ret("acl.rt.malloc_host", ret)
                ret = acl.rt.memcpy(output_host,
                                    infer_output_size,
                                    infer_output_ptr,
                                    infer_output_size,
                                    ACL_MEMCPY_DEVICE_TO_HOST)
                check_ret("acl.rt.memcpy", ret)
                dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, i)
                check_ret("acl.mdl.get_cur_output_dims", ret)
                ptr = output_host
                if "ptr_to_bytes" in dir(acl.util):
                    bytes_data = acl.util.ptr_to_bytes(ptr, infer_output_size)
                    data = np.frombuffer(bytes_data, dtype=self.output_numpy_dtypes[i]).reshape(self.output_dims[i])
                else:
                    data = acl.util.ptr_to_numpy(ptr, self.output_dims[i],
                                                 acl_dtype_idx_to_numpy_dtype_idx[self.output_acl_dtype_idxs[i]])
                current_inference_output.append(data.copy())
                acl.rt.free_host(output_host)
            self._destroy_dataset(input_dataset_ptr)
            self._destroy_dataset(output_dataset_ptr)
            inference_outputs.append(current_inference_output)
        return inference_outputs

    @require_npu_context
    def synchronize_device(self):
        acl.rt.synchronize_device(self.device_id)


# Exposed to outer of package
class ACLNetHandler(object):

    def __init__(self, npu_device_ids: Union[List[int], int], om_model_path: str):
        # print(om_model_path)
        self.npu_device_ids = [npu_device_ids] if isinstance(npu_device_ids, int) else npu_device_ids
        if len(self.npu_device_ids) == 0:
            raise RuntimeError('no NPU specified')
        if len(self.npu_device_ids) > len(set(self.npu_device_ids)):
            raise RuntimeError('duplicated NPU device IDs')
        self.acl_net = [Net(i, om_model_path) for i in self.npu_device_ids]
        self.batch_size = self.acl_net[0].get_input_batch_size()

    def forward(self, images_data: np.ndarray) -> Union[List[np.ndarray], np.ndarray]:
        real_batch_size: int = images_data.shape[0]
        rem: int = real_batch_size % self.batch_size
        if rem != 0:
            padding_config = tuple(
                (0, self.batch_size - rem) if i == 0 else (0, 0) for i in range(len(images_data.shape)))
            # Pad zeros along the 0th axis
            images_data = np.pad(images_data, padding_config, mode='constant')
        jobs = [images_data[i:i + self.batch_size] for i in range(0, images_data.shape[0], self.batch_size)]
        jobs_num = len(jobs)
        real_npu_num = min(len(self.acl_net), jobs_num)
        q: int = jobs_num // real_npu_num
        r: int = jobs_num % real_npu_num
        dispatched_jobs: List[List[np.ndarray]] = []
        idx = 0
        for i in range(real_npu_num):
            nxt: int = idx + q + (1 if i < r else 0)
            dispatched_jobs.append(jobs[idx:nxt])
            idx = nxt
        # print("dispatch_jobs: ", just_get_shape(dispatched_jobs))
        result = self._forward(dispatched_jobs)

        for i in range(len(result)):
            result[i] = result[i][:real_batch_size, ...]
        if len(result) == 1:
            result = result[0]
        return result

    def _forward(self, images_batches_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        real_npu_num = len(images_batches_list)

        for i in range(real_npu_num):
            self.acl_net[i].dispatch_parallel_job(images_batches_list[i])
        tmp = self.acl_net[0].fetch_output_from_npu()
        output: List[List[np.ndarray]] = [list(row) for row in zip(*tmp)]
        for i in range(1, real_npu_num):
            for batch_outputs in self.acl_net[i].fetch_output_from_npu():
                for output_idx, batch_output in enumerate(batch_outputs):
                    output[output_idx].append(batch_output)
        for i in range(real_npu_num):
            self.acl_net[i].synchronize_device()
        # print("_forward: {}".format(just_get_shape(output)))
        stacked_output = [np.concatenate(o, axis=0) for o in output]
        return stacked_output

    def __call__(self, batch_images: Union[List[np.ndarray], np.ndarray]) \
            -> Union[List[np.ndarray], np.ndarray]:
        return self.forward(batch_images)

    def __del__(self):
        while len(self.acl_net) > 0:
            del self.acl_net[-1]

    def release(self):
        self.__del__()


def just_get_shape(x: Union[List, np.ndarray]):
    if isinstance(x, list):
        return [just_get_shape(i) for i in x]
    return x.shape


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