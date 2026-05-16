import numpy as np
from PIL import Image
from pathlib import Path


def image_level_grid_occlusion(
    image: np.ndarray,
    rho_grids: int = 20,
    line_width: int = 1
) -> np.ndarray:
    """
    Apply grid-based occlusion attack to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image in HWC format.

    rho_grids : int
        Grid density.

    line_width : int
        Width of grid lines.

    Returns
    -------
    np.ndarray
        Occluded image.
    """

    attacked = image.copy()

    h, w = attacked.shape[:2]

    vertical_positions = np.linspace(
        0,
        w - 1,
        rho_grids,
        dtype=int
    )

    for x in vertical_positions:
        x_start = max(0, x - line_width // 2)
        x_end = min(w, x + line_width // 2 + 1)

        attacked[:, x_start:x_end] = 0

    horizontal_positions = np.linspace(
        0,
        h - 1,
        rho_grids,
        dtype=int
    )

    for y in horizontal_positions:
        y_start = max(0, y - line_width // 2)
        y_end = min(h, y + line_width // 2 + 1)

        attacked[y_start:y_end, :] = 0

    return attacked



def grid_attack(input_path: str = None, output_bool: bool = False, rho_grids: int = 25, line_width: int = 1) -> Image.Image:
    """
    Applies grid-based occlusion attack to an image and optionally saves the result.

    Parameters
    ----------
    input_path : str
        Path to the input image.
    output_bool : bool
        Whether to save the attacked image.
    rho_grids : int
        Grid density.
    line_width : int
        Width of grid lines.

    Returns
    -------
    Image.Image
        Attacked image.
    """
    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img)
    attacked = image_level_grid_occlusion(
        img_np,
        rho_grids=rho_grids,
        line_width=line_width
    )
    attacked_img = Image.fromarray(attacked)
    if output_bool:
        repo_root = Path(__file__).resolve().parent.parent.parent
        output_path = repo_root / "results" / Path(input_path).name.replace(".jpg", "_grid_attack.jpg")
        attacked_img.save(output_path)
    return attacked_img
    
if __name__ == "__main__":
    attacked_img = grid_attack( "samples/person4.jpg", output_bool=True, rho_grids=25, line_width=1)
    attacked_img.show()

    