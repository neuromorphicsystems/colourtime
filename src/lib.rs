#[pyo3::pyfunction]
fn stack(
    base: &numpy::PyArray3<f64>,
    xy: numpy::PyReadonlyArray2<u16>,
    colours: numpy::PyReadonlyArray2<f64>,
    alpha: f64,
) -> pyo3::PyResult<()> {
    let base_shape = base.shape();
    if base_shape.len() != 3 || base_shape[2] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "base must have shape (w, h, 4)",
        ));
    }
    let xy_shape = xy.shape();
    if xy_shape.len() != 2 || xy_shape[1] != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "xy must have shape (n, 2)",
        ));
    }
    let colours_shape = colours.shape();
    if colours_shape.len() != 2 || colours_shape[1] != 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "colours must have shape (n, 4)",
        ));
    }
    if xy_shape[0] != colours_shape[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "xy and colours must have the same number of rows",
        ));
    }
    if alpha <= 0.0 || alpha > 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "alpha must be in the range ]0, 1]",
        ));
    }

    // safety: Calling this method invalidates all other references to the internal data
    let mut base_view = unsafe { base.as_array_mut() };

    let xy_view = xy.as_array();
    let colours_view = colours.as_array();
    for (xy, colour) in std::iter::zip(xy_view.rows(), colours_view.rows()) {
        let x = xy[0] as usize;
        let y = xy[1] as usize;
        base_view[[x, y, 0]] = (1.0 - alpha) * base_view[[x, y, 0]] + alpha * colour[0];
        base_view[[x, y, 1]] = (1.0 - alpha) * base_view[[x, y, 1]] + alpha * colour[1];
        base_view[[x, y, 2]] = (1.0 - alpha) * base_view[[x, y, 2]] + alpha * colour[2];
        base_view[[x, y, 3]] = (1.0 - alpha) * base_view[[x, y, 3]] + alpha * colour[3];
    }
    Ok(())
}

#[pyo3::pymodule]
fn colourtime_extension(_py: pyo3::Python<'_>, module: &pyo3::types::PyModule) -> pyo3::PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(stack, module)?)?;
    Ok(())
}
