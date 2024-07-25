use rand::Rng;
use crate::ValueWrapper;

// Module trait (equivalent to Python's Module class)
pub trait Module {
    fn zero_grad(&mut self);
    fn parameters(&self) -> Vec<ValueWrapper>;
}

pub struct Neuron {
    w: Vec<ValueWrapper>,
    b: ValueWrapper,
    nonlin: bool,
}

impl Neuron {
    pub fn new(nin: usize, nonlin: bool) -> Self {
        let mut rng = rand::thread_rng();
        Neuron {
            w: (0..nin)
                .map(|_| ValueWrapper::new(rng.gen_range(-1.0..1.0)))
                .collect(),
            b: ValueWrapper::new(0.0),
            nonlin,
        }
    }

    pub fn call(&self, x: &[ValueWrapper]) -> ValueWrapper {
        let mut act = self.b.clone();
        for (wi, xi) in self.w.iter().zip(x.iter()) {
            act += wi.clone() * xi.clone();
        }
        if self.nonlin {
            act.relu()
        } else {
            act
        }
    }
}

impl Module for Neuron {
    fn zero_grad(&mut self) {
        for w in &self.w {
            w.0.borrow_mut().grad = 0.0;
        }
        self.b.0.borrow_mut().grad = 0.0;
    }

    fn parameters(&self) -> Vec<ValueWrapper> {
        let mut params = self.w.clone();
        params.push(self.b.clone());
        params
    }
}


impl std::fmt::Debug for Neuron {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}Neuron({})", if self.nonlin { "ReLU" } else { "Linear" }, self.w.len())
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonlin: bool) -> Self {
        Layer {
            neurons: (0..nout).map(|_| Neuron::new(nin, nonlin)).collect(),
        }
    }

    pub fn call(&self, x: &[ValueWrapper]) -> Vec<ValueWrapper> {
        self.neurons.iter().map(|n| n.call(x)).collect()
    }
}

impl Module for Layer {
    fn zero_grad(&mut self) {
        for neuron in &mut self.neurons {
            neuron.zero_grad();
        }
    }

    fn parameters(&self) -> Vec<ValueWrapper> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

impl std::fmt::Debug for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Layer of [")?;
        for (i, neuron) in self.neurons.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", neuron)?;
        }
        write!(f, "]")
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: &[usize]) -> Self {
        let mut sizes = vec![nin];
        sizes.extend_from_slice(nouts);

        let layers = (0..nouts.len())
            .map(|i| Layer::new(sizes[i], sizes[i+1], i != nouts.len() - 1))
            .collect();

        MLP { layers }
    }

    pub fn call(&self, mut x: Vec<ValueWrapper>) -> Vec<ValueWrapper> {
        for layer in &self.layers {
            x = layer.call(&x);
        }
        x
    }
}

impl Module for MLP {
    fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }

    fn parameters(&self) -> Vec<ValueWrapper> {
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }
}

impl std::fmt::Debug for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MLP of [")?;
        for (i, layer) in self.layers.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", layer)?;
        }
        write!(f, "]")
    }
}