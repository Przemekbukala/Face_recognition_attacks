import numpy as np
from PIL import Image
from pathlib import Path


import numpy as np

def image_level_grid_occlusion(
    image: np.ndarray,
    rho_grids: int = 20,
    line_width: int = 1,
    random_spacing: bool = False,
    diag_lines: bool = False
) -> np.ndarray:
    """
    Apply grid-based occlusion attack to an image.

    Parameters
    ----------
    image : np.ndarray
        Input image in HWC format.

    rho_grids : int
        Grid density (number of grid lines).

    line_width : int
        Width of grid lines.

    random_spacing : bool
        If True, grid lines are randomly spaced. If False, they are spaced evenly.

    diag_lines : bool
        If True, occlusion will include only diagonal grid lines.

    Returns
    -------
    np.ndarray
        Occluded image.
    """
    attacked = image.copy()
    h, w = attacked.shape[:2]

    if diag_lines:
        if random_spacing:
            offsets = np.sort(
                np.random.choice(
                    np.arange(-h + 1, w), size=min(rho_grids, h + w - 1), replace=False
                )
            )
        else:
            offsets = np.linspace(-h + 1, w - 1, rho_grids, dtype=int)

        for offset in offsets:
            for i in range(h):
                j = offset + i
                if 0 <= j < w:
                    y_start = max(0, i - line_width // 2)
                    y_end = min(h, i + line_width // 2 + 1)
                    x_start = max(0, j - line_width // 2)
                    x_end = min(w, j + line_width // 2 + 1)
                    attacked[y_start:y_end, x_start:x_end] = 0

        if random_spacing:
            offsets = np.sort(
                np.random.choice(
                    np.arange(0, h + w - 1), size=min(rho_grids, h + w - 1), replace=False
                )
            )
        else:
            offsets = np.linspace(0, h + w - 2, rho_grids, dtype=int)

        for offset in offsets:
            for i in range(h):
                j = offset - i
                if 0 <= j < w:
                    y_start = max(0, i - line_width // 2)
                    y_end = min(h, i + line_width // 2 + 1)
                    x_start = max(0, j - line_width // 2)
                    x_end = min(w, j + line_width // 2 + 1)
                    attacked[y_start:y_end, x_start:x_end] = 0

    else:
        if random_spacing:
            vertical_positions = np.sort(
                np.random.choice(np.arange(w), size=min(rho_grids, w), replace=False)
            )
            horizontal_positions = np.sort(
                np.random.choice(np.arange(h), size=min(rho_grids, h), replace=False)
            )
        else:
            vertical_positions = np.linspace(0, w - 1, rho_grids, dtype=int)
            horizontal_positions = np.linspace(0, h - 1, rho_grids, dtype=int)

        for x in vertical_positions:
            x_start = max(0, x - line_width // 2)
            x_end = min(w, x + line_width // 2 + 1)
            attacked[:, x_start:x_end] = 0

        for y in horizontal_positions:
            y_start = max(0, y - line_width // 2)
            y_end = min(h, y + line_width // 2 + 1)
            attacked[y_start:y_end, :] = 0

    return attacked



def grid_attack(
    input_path: str = None, 
    output_bool: bool = False, 
    rho_grids: int = 25, 
    line_width: int = 1, 
    random_spacing: bool = False, 
    diag_lines: bool = False
    ) -> Image.Image:
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
    random_spacing : bool
        Whether to use random spacing for grid lines.
    diag_lines : bool
        If True, occlusion will include only diagonal grid lines.

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
        line_width=line_width,
        random_spacing=random_spacing,
        diag_lines=diag_lines
    )
    attacked_img = Image.fromarray(attacked)
    if output_bool:
        repo_root = Path(__file__).resolve().parent.parent.parent
        output_path = repo_root / "results" / Path(input_path).name.replace(".jpg", "_grid_attack.jpg")
        attacked_img.save(output_path)
    return attacked_img
    
if __name__ == "__main__":
    attacked_img = grid_attack( "samples/person3.jpg", output_bool=True, rho_grids=20, line_width=1, random_spacing=True, diag_lines=True)
    attacked_img.show()

    