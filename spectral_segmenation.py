import cv2
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def show(img):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.imshow(img)
    plt.show()

def save_binary_image(
    binary_array,
    output_path,
    description="",
    mm_per_pixel=8.02,
    dpi=150
):
    """
    Save a binary image with axes scaled to physical units (mm),
    and description placed outside the image as a title.

    Parameters:
        binary_array (np.ndarray): 2D binary array
        output_path (str): Output image path
        description (str): Text to display outside the image (title)
        mm_per_pixel (float): Physical size of one pixel in mm
        dpi (int): Image resolution
    """

    h, w = binary_array.shape

    # Physical dimensions
    width_mm  = w * mm_per_pixel
    height_mm = h * mm_per_pixel

    fig, ax = plt.subplots()

    # Map pixels to physical coordinates
    ax.imshow(
        binary_array,
        cmap="gray",
        vmin=0,
        vmax=1,
        extent=[0, width_mm, height_mm, 0]  # origin at top-left
    )

    # Label axes
    ax.set_xlabel("μm")
    ax.set_ylabel("μm")
    ax.set_aspect("equal")

    # Place description as title
    if description:
        ax.set_title(description, fontsize=12, pad=15)

    # Save figure
    fig.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=dpi
    )
    plt.close(fig)




def lo_pass_filter(img, radius=17):
    """ 
        img: greyscale image (ndarray)
    """
    ft = np.fft.fft2(img)
    ftshift = np.fft.fftshift(ft)
    spectre = 20*np.log(np.abs(ftshift))

    x, y = np.indices(img.shape)
    center = (img.shape[0]//2, img.shape[1]//2)
    mask = (x - center[0])**2 + (y - center[1])**2 < radius**2

    ftshift_mask = ftshift*(mask)
    ftishift_back = np.fft.ifftshift(ftshift_mask)
    img_back = np.fft.ifft2(ftishift_back)
    img_back = np.abs(img_back)

    return img_back

def segment(img_path:Path, out_path: Path, out_file:Path, pixel_size_um=8.02, 
            threshold=100, maxval=120, lo_pass_radius=18):
    img = cv2.imread(str(img_p))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_back = lo_pass_filter(img, radius=lo_pass_radius)
    
    threshod, seg = cv2.threshold(img_back, 100,120, cv2.THRESH_BINARY_INV)
    seg_binary = seg > threshold

    # pixel counts
    area = np.sum(seg_binary)
    lx = np.sum(seg_binary, axis=0, dtype=bool).sum()
    ly = np.sum(seg_binary, axis=1, dtype=bool).sum()
    
    # in micro meters
    area_m = area * pixel_size_um
    lx_m = lx * pixel_size_um
    ly_m = ly * pixel_size_um
    stats_str = f"area = {area_m:.2f}μm,  ly = {ly_m:.2f}μm,  lx = {lx_m:.2f}μm"
    print(f"----- {img_path.name} segmentation: ------")
    print(stats_str)
    
    with open(out_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([str(img_p), area_m, lx_m, ly_m])

    # save
    save_binary_image(seg_binary, 
                      out_path, 
                      description=f"{img_path.name} | "+stats_str)

if __name__=='__main__':
    in_p = 'data/podl3_rub'
    out_dir = 'results'
    lo_pass_radius = 17

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    out_file = 'results.csv'

    fields=['input_file', 'area','lx','ly']
    with open(out_dir / out_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    

    for extension in {"jp2", "tif", "jpg", "jpeg"}:
        for img_p in Path(in_p).glob(f"*.{extension}"):
            out_p = out_dir / f"{img_p.stem}_seg.png"
            print(f"input: {img_p}, output: {out_p}")
            segment(img_p, out_p, out_dir / out_file, lo_pass_radius=lo_pass_radius)