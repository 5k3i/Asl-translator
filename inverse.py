import cv2, sys, uuid, os

has_args = False

if len(sys.argv) > 1:
    has_args = True

def print_help() -> None:
    pass

# invert filter function
def invert_photo(image: cv2.typing.MatLike, angle: int) -> None:
    assert image is not None, "file could not be read, check with os.path.exists()"
    rows,cols = image.shape
    # cols-1 and rows-1 are the coordinate limits.
    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
    dst = cv2.warpAffine(image,M,(cols,rows))

cv2.namedWindow("window")
image = cv2.imread("Test-Folder")



