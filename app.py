import argparse
import logging
import re
from functools import partial
from http import HTTPStatus
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, Query, UploadFile
from nougat import NougatModel
from nougat.postprocessing import markdown_compatible
from nougat.utils.args import get_common_args
from nougat.utils.checkpoint import get_checkpoint
from nougat.utils.dataset import LazyDataset
from nougat.utils.device import move_to_device
from pydantic import BaseModel
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import pypdf

logging.basicConfig(level=logging.INFO)

model = None
app = FastAPI()


class PDFResponse(BaseModel):
    content: str


class HealthCheckResponse(BaseModel):
    status_code: int
    data: dict


def get_args():
    parser = argparse.ArgumentParser()
    parser = get_common_args(parser)
    parser.add_argument("--port", type=int, default=8503, help="Port for the API server.")
    args = parser.parse_args()
    if args.checkpoint is None or not args.checkpoint.exists():
        # args.checkpoint = get_checkpoint(args.checkpoint, model_tag=args.model)
        args.checkpoint = get_checkpoint(args.checkpoint, model_tag="0.1.0-base")
    if args.batchsize <= 0:
        args.batchsize = 1
    return args


def process_pdf(pdf_path, pages=None):
    if not pdf_path.exists():
        return None
    try:
        dataset = LazyDataset(
            str(pdf_path.resolve()),
            partial(model.encoder.prepare_input, random_padding=False),
            pages,
        )
    except pypdf.errors.PdfStreamError:
        logging.info(f"Could not load file {str(pdf_path)}.")
        return None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate,
    )
    predictions = []
    page_num = 0
    for i, (sample, is_last_page) in enumerate(tqdm(dataloader)):
        model_output = model.inference(
            image_tensors=sample, early_stopping=args.skipping
        )
        for j, output in enumerate(model_output["predictions"]):
            if page_num == 0:
                logging.info(
                    "Processing file %s with %i pages" % (pdf_path.name, dataset.size)
                )
            page_num += 1
            if output.strip() == "[MISSING_PAGE_POST]":
                predictions.append(f"\n\n[MISSING_PAGE_EMPTY:{page_num}]\n\n")
            elif args.skipping and model_output["repeats"][j] is not None:
                if model_output["repeats"][j] > 0:
                    logging.warning(f"Skipping page {page_num} due to repetitions.")
                    predictions.append(f"\n\n[MISSING_PAGE_FAIL:{page_num}]\n\n")
                else:
                    predictions.append(
                        f"\n\n[MISSING_PAGE_EMPTY:{i*args.batchsize+j+1}]\n\n"
                    )
            else:
                if args.markdown:
                    output = markdown_compatible(output)
                predictions.append(output)
    out = "".join(predictions).strip()
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


@app.get("/", response_model=HealthCheckResponse)
def health_check():
    """Health check endpoint."""
    response = {
        "status_code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict", response_model=PDFResponse)
async def predict(
    file: UploadFile = File(...),
    pages: str = Query(None),
):
    pdf_path = Path(file.filename)
    with pdf_path.open("wb") as f:
        f.write(await file.read())
    if pages:
        pages = [int(p) - 1 for p in pages.split(",")]
    result = process_pdf(pdf_path, pages)
    pdf_path.unlink()
    if result is None:
        return {"content": "Failed to process PDF."}
    return {"content": result}


def main():
    global args, model
    args = get_args()
    model = NougatModel.from_pretrained(args.checkpoint)
    model = move_to_device(model, bf16=not args.full_precision, cuda=args.batchsize > 0)
    model.eval()
    import uvicorn
    uvicorn.run(app, port=args.port)


if __name__ == "__main__":
    main()
