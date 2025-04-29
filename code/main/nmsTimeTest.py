from lib.config import ATKEDIMG_PATH,ORIGINIMG_PATH
from lib.output import get_raw_output_img
from lib.nms import run_nms_on_preds

preds, im_tensor = get_raw_output_img(ATKEDIMG_PATH)
run_nms_on_preds(preds)
preds, im_tensor = get_raw_output_img(ORIGINIMG_PATH)
run_nms_on_preds(preds)