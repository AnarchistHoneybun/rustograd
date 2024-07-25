use crate::engine::ValueWrapper;

#[test]
fn sanity_check() {
    let x = ValueWrapper::new(-4.0);
    let z = 2.0 * x.clone() + 2.0 + x.clone();
    let q = z.clone().relu() + z.clone() * x.clone();
    let h = (z.clone() * z.clone()).relu();
    let mut y = h.clone() + q.clone() + q.clone() * x.clone();
    y.backward();

    assert_eq!(y.0.borrow().data, -20.0);
    assert_eq!(x.0.borrow().grad, 46.0);
}

#[test]
fn more_ops() {
    let a = ValueWrapper::new(-4.0);
    let b = ValueWrapper::new(2.0);
    let mut c = a.clone() + b.clone();
    let mut d = a.clone() * b.clone() + b.clone().pow(3.0);
    c += c.clone() + 1.0;
    c += 1.0 + c.clone() + (-a.clone());
    d += d.clone() * 2.0 + (b.clone() + a.clone()).relu();
    d += 3.0 * d.clone() + (b.clone() - a.clone()).relu();
    let e = c.clone() - d.clone();
    let f = e.clone().pow(2.0);
    let mut g = f.clone() / 2.0;
    g += 10.0 / f;
    g.backward();

    let expected_a_grad = 138.83381924198252 /* expected a.grad from Python */;
    let expected_b_grad = 645.5772594752186 /* expected b.grad from Python */;
    let expected_g_data = 24.70408163265306 /* expected g.data from Python */;
    let tol = 1e-6;

    assert!((g.0.borrow().data - expected_g_data).abs() < tol);
    assert!((a.0.borrow().grad - expected_a_grad).abs() < tol);
    assert!((b.0.borrow().grad - expected_b_grad).abs() < tol);
}
