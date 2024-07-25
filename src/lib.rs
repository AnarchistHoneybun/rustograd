use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

#[cfg(test)]
mod tests;

pub mod nn;

#[derive(Clone)]
enum Op {
    Add(Rc<RefCell<Value>>, Rc<RefCell<Value>>),
    Mul(Rc<RefCell<Value>>, Rc<RefCell<Value>>),
    Pow(Rc<RefCell<Value>>, f64),
    Relu(Rc<RefCell<Value>>),
    Leaf,
}

#[derive(Clone)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
    op: Op,
}

impl Value {
    pub fn new(data: f64) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            op: Op::Leaf,
        }))
    }
}

#[derive(Clone)]
pub struct ValueWrapper(pub Rc<RefCell<Value>>);

impl ValueWrapper {
    pub fn new(data: f64) -> Self {
        ValueWrapper(Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            op: Op::Leaf,
        })))
    }
}

impl fmt::Display for ValueWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = self.0.borrow();
        write!(f, "Value(data={}, grad={})", value.data, value.grad)
    }
}

impl ValueWrapper {
    pub fn backward(&mut self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        build_topo(Rc::clone(&self.0), &mut topo, &mut visited);

        self.0.borrow_mut().grad = 1.0;

        for v in topo.iter().rev() {
            let v_grad = v.borrow().grad;
            match &v.borrow().op {
                Op::Add(a, b) => {
                    a.borrow_mut().grad += v_grad;
                    b.borrow_mut().grad += v_grad;
                },
                Op::Mul(a, b) => {
                    let a_data = a.borrow().data;
                    let b_data = b.borrow().data;
                    {
                        let mut a = a.borrow_mut();
                        a.grad += b_data * v_grad;
                    }
                    {
                        let mut b = b.borrow_mut();
                        b.grad += a_data * v_grad;
                    }
                },
                Op::Pow(a, power) => {
                    let a_data = a.borrow().data;
                    {
                        let mut a = a.borrow_mut();
                        a.grad += power * a_data.powf(power - 1.0) * v_grad;
                    }
                },
                Op::Relu(a) => {
                    if v.borrow().data > 0.0 {
                        a.borrow_mut().grad += v_grad;
                    }
                },
                Op::Leaf => {}
            }
        }
    }
}

impl Add for ValueWrapper {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let result = Rc::new(RefCell::new(Value {
            data: self.0.borrow().data + other.0.borrow().data,
            grad: 0.0,
            op: Op::Add(Rc::clone(&self.0), Rc::clone(&other.0)),
        }));
        ValueWrapper(result)
    }
}

impl Add<f64> for ValueWrapper {
    type Output = Self;

    fn add(self, other: f64) -> Self {
        self + ValueWrapper::new(other)
    }
}

impl Add<ValueWrapper> for f64 {
    type Output = ValueWrapper;

    fn add(self, other: ValueWrapper) -> ValueWrapper {
        ValueWrapper::new(self) + other
    }
}

impl AddAssign for ValueWrapper {
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl AddAssign<f64> for ValueWrapper {
    fn add_assign(&mut self, other: f64) {
        *self = self.clone() + ValueWrapper::new(other);
    }
}

impl Mul for ValueWrapper {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let result = Rc::new(RefCell::new(Value {
            data: self.0.borrow().data * other.0.borrow().data,
            grad: 0.0,
            op: Op::Mul(Rc::clone(&self.0), Rc::clone(&other.0)),
        }));
        ValueWrapper(result)
    }
}

impl Mul<f64> for ValueWrapper {
    type Output = Self;

    fn mul(self, other: f64) -> Self {
        self * ValueWrapper::new(other)
    }
}

impl Mul<ValueWrapper> for f64 {
    type Output = ValueWrapper;

    fn mul(self, other: ValueWrapper) -> ValueWrapper {
        ValueWrapper::new(self) * other
    }
}

impl Neg for ValueWrapper {
    type Output = Self;

    fn neg(self) -> Self {
        self * -1.0
    }
}

impl Sub for ValueWrapper {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl Sub<f64> for ValueWrapper {
    type Output = Self;

    fn sub(self, other: f64) -> Self {
        self + (-ValueWrapper::new(other))
    }
}

impl Sub<ValueWrapper> for f64 {
    type Output = ValueWrapper;

    fn sub(self, other: ValueWrapper) -> ValueWrapper {
        ValueWrapper::new(self) + (-other)
    }
}

impl ValueWrapper {
    pub fn pow(&self, power: f64) -> Self {
        let result = Rc::new(RefCell::new(Value {
            data: self.0.borrow().data.powf(power),
            grad: 0.0,
            op: Op::Pow(Rc::clone(&self.0), power),
        }));
        ValueWrapper(result)
    }
}

impl Div for ValueWrapper {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        self * other.pow(-1.0)
    }
}

impl Div<f64> for ValueWrapper {
    type Output = Self;

    fn div(self, other: f64) -> Self {
        self / ValueWrapper::new(other)
    }
}

impl Div<ValueWrapper> for f64 {
    type Output = ValueWrapper;

    fn div(self, other: ValueWrapper) -> ValueWrapper {
        ValueWrapper::new(self) / other
    }
}

impl ValueWrapper {
    pub fn relu(&self) -> Self {
        let result = Rc::new(RefCell::new(Value {
            data: self.0.borrow().data.max(0.0),
            grad: 0.0,
            op: Op::Relu(Rc::clone(&self.0)),
        }));
        ValueWrapper(result)
    }
}

fn build_topo(v: Rc<RefCell<Value>>, topo: &mut Vec<Rc<RefCell<Value>>>, visited: &mut HashSet<*mut Value>) {
    let ptr = v.as_ptr();
    if !visited.contains(&ptr) {
        visited.insert(ptr);
        match &v.borrow().op {
            Op::Add(a, b) | Op::Mul(a, b) => {
                build_topo(Rc::clone(a), topo, visited);
                build_topo(Rc::clone(b), topo, visited);
            },
            Op::Pow(a, _) | Op::Relu(a) => {
                build_topo(Rc::clone(a), topo, visited);
            },
            Op::Leaf => {}
        }
        topo.push(Rc::clone(&v));
    }
}