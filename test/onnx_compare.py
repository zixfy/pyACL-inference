import sys, os
import numpy as np
import onnxruntime as ort
from huawei_api import ACLNetHandler, using_huawei_api


def cos_sim(v1, v2):
    return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

@using_huawei_api
def main(bsn = 5):
    huawei_model = ACLNetHandler(npu_device_ids=[0, 1],om_model_path="/root/disk/zzx/huawei-npu/transreid_bs16_highprecision.om")
    onnx_model = ort.InferenceSession("/root/disk/torch_program/model/pth2om/transreid_bs16.onnx")
    bsz = 16
    rbsz = bsz * bsn
    dummmy = np.random.random((rbsz, 3, 256, 128)).astype(np.float32)
    input_name = onnx_model.get_inputs()[0].name
    onnx_out = [onnx_model.run([], {input_name: dummmy[i * bsz:(i + 1) * bsz]})[0] for i in range(bsn)]
    onnx_out = np.concatenate(onnx_out, axis=0)
    om_out = huawei_model(dummmy)
    print("cls token cos_sim: ", cos_sim(onnx_out[:, 0, :].flatten(), om_out[:, 0, :].flatten()))
    huawei_model.release()

main()
