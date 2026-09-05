#!/usr/bin/env python3
"""Read ZIP directories and TIFF headers without decoding image pixels."""
import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from PIL import Image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError("Preserve previous inventory")
    rows=[]
    for path in sorted(args.directory.glob("tiffs_*eye_lenses*.zip")):
        with zipfile.ZipFile(path) as archive:
            entries=[i for i in archive.infolist() if not i.is_dir()]
            shape_modes=set();decoded_bytes=0;physical=[];ids=[]
            for entry in entries:
                match=re.search(r"(\d+)\.tif$",entry.filename,re.I)
                if not match:raise ValueError("Unexpected archive member")
                ids.append(int(match[1]))
                with Image.open(io.BytesIO(archive.read(entry))) as im:
                    shape_modes.add((im.width,im.height,im.mode))
                    if im.mode!="P" or im.tag_v2.get(258)!=(8,):
                        raise ValueError("Decoded byte estimate expects 8-bit palette indices")
                    decoded_bytes+=im.width*im.height
                    if any(k in im.tag_v2 for k in (270,282,283)) or im.tag_v2.get(296,1)!=1:
                        physical.append({"member":entry.filename,"tags":{str(k):str(im.tag_v2[k]) for k in (270,282,283,296) if k in im.tag_v2}})
            ids.sort()
            rows.append({"archive":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                         "zip_bytes":path.stat().st_size,"tiff_file_bytes_after_unzip":sum(i.file_size for i in entries),
                         "decoded_uint8_bytes":decoded_bytes,"slices":len(entries),
                         "slice_number_first":ids[0],"slice_number_last":ids[-1],
                         "contiguous_unique_slice_numbers":ids==list(range(ids[0],ids[-1]+1)),
                         "image_shapes_modes":sorted(shape_modes),"possible_calibration_metadata":physical})
    result={"inspection":"All TIFF headers; no pixel decoding in this inventory",
            "available_archives":len(rows),"total_decoded_uint8_bytes":sum(r["decoded_uint8_bytes"] for r in rows),
            "failed_attachment_not_inspected":"tiffs_M3_F_28_03_eye_lenses-20260903T135112Z-1-001(1).zip",
            "archives":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"archives":len(rows),"total_decoded_GB":result["total_decoded_uint8_bytes"]/1e9,
                      "files_with_possible_calibration":sum(bool(r["possible_calibration_metadata"]) for r in rows)}))


if __name__=="__main__":main()
