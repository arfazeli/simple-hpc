"""Cell-centre coordinates and centreline extraction."""
import numpy as np


def cell_coords(NX, NY):
    """Return (x, y): 1D cell-centre coordinates.

    x has NY entries (columns), y has NX entries (rows), y increasing with
    index, i.e. y[k] is the height of row NX-1-k (row 0 is the lid).
    """
    x = (np.arange(NY) + 0.5) / NY
    y = (np.arange(NX) + 0.5) / NX
    return x, y


def _mid(a, axis):
    """Average the two middle entries along axis if even, else take the middle."""
    n = a.shape[axis]
    if n % 2 == 0:
        return 0.5 * (np.take(a, n // 2 - 1, axis=axis) + np.take(a, n // 2, axis=axis))
    return np.take(a, n // 2, axis=axis)


def centrelines(run):
    """Return dict(y, u, x, v) including wall points, sorted by increasing coordinate."""
    info = run["info"]
    NX, NY = int(info["N_cells_x"]), int(info["N_cells_y"])
    x, y = cell_coords(NX, NY)
    # u along x = 0.5: column mid, rows top->bottom; flip so y increases
    u_col = _mid(run["ux_centre"], 1)[::-1]
    v_row = _mid(run["uy_centre"], 0)
    yy = np.concatenate(([0.0], y, [1.0]))
    uu = np.concatenate(([0.0], u_col, [1.0]))
    xx = np.concatenate(([0.0], x, [1.0]))
    vv = np.concatenate(([0.0], v_row, [0.0]))
    return dict(y=yy, u=uu, x=xx, v=vv)


if __name__ == "__main__":
    x, y = cell_coords(4, 4)
    ref = np.array([0.125, 0.375, 0.625, 0.875])
    assert np.allclose(x, ref) and np.allclose(y, ref)
    ux = np.tile(np.arange(4.0), (4, 1))  # value = column index
    run = {"info": {"N_cells_x": 4, "N_cells_y": 4}, "ux_centre": ux, "uy_centre": ux.T.copy()}
    c = centrelines(run)
    assert np.allclose(c["u"][1:-1], 1.5), "u must average columns 1 and 2"
    assert np.allclose(c["v"][1:-1], 1.5), "v must average rows 1 and 2"
    assert c["u"][0] == 0 and c["u"][-1] == 1 and c["v"][0] == 0 and c["v"][-1] == 0
    print("fields self-check OK")
