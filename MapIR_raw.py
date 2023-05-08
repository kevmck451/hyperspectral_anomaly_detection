#! /usr/bin/env nix-shell
#! nix-shell -i python3 -p "python3.withPackages (ps: [ ps.numpy ps.opencv4 ])"
#! nix-shell -I nixpkgs=https://github.com/NixOS/nixpkgs/archive/8c66bd1b68f4708c90dcc97c6f7052a5a7b33257.tar.gz

import numpy as np
import cv2
import argparse

# derived from info in
# https://github.com/mapircamera/camera-scripts/tree/main/Convert_Survey3_RAW_To_Tiff

def mapir_unpack(data):
    # unpack a mapir raw image file into per-pixel 12 bit values

    assert len(data) == 1.5*4000*3000

    # two pixels are packed into each 3 byte value
    # ABC = nibbles of even pixel (big endian)
    # DEF = nibbles of odd pixel (big endian)
    # bytes in data: BC FA DE

    data = np.frombuffer(data, dtype=np.uint8).astype(np.uint16)

    # pull the first of every third byte as the even pixel data
    even_pixels = data[0::3]
    # pull the last of every third byte as the odd pixel data
    odd_pixels = data[2::3]
    # the middle byte has bits of both pixels
    middle_even = data[1::3].copy()
    middle_even &= 0xF
    middle_even <<= 8

    middle_odd = data[1::3].copy()
    middle_odd &= 0xF0
    middle_odd >>= 4
    odd_pixels <<= 4

    # combine middle byte data into pixel data
    even_pixels |= middle_even
    odd_pixels |= middle_odd

    pixels = np.stack((even_pixels, odd_pixels), axis=-1)

    # reshape to form camera image 4000x3000 pixels
    image = pixels.reshape((3000, 4000))

    return image

def mapir_debayer(data):
    # the bayer pattern is
    # B G
    # G R

    # use opencv's debayering routine on the data
    return cv2.cvtColor(data, cv2.COLOR_BAYER_RG2RGB)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("raw_in", type=str,
        help="Path to input raw file.")
    parser.add_argument("img_out", type=str,
        help="Path to output image file.")

    args = parser.parse_args()

    with open(args.raw_in, "rb") as f:
        data = f.read()

    unpacked_data = mapir_unpack(data)
    debayered_data = mapir_debayer(unpacked_data)

    # convert to 8 bit for image output
    output_data = (debayered_data >> 4).astype(np.uint8)

    cv2.imwrite(args.img_out, output_data)

if __name__ == "__main__":
    main()
