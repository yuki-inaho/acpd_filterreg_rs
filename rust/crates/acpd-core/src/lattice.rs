//! Genuine permutohedral Splat/Blur/Slice, not radius truncation.
//!
//! Algorithmic port of the BSD-3-Clause code in the supplied probreg archive.
//! Original copyright (c) 2013 Philipp Kraehenbuehl; see THIRD_PARTY_NOTICES.md.
//! f64 arithmetic and i64 keys replace the original float/short representation.
use crate::types::*;
use std::collections::HashMap;
pub type LatticeKey = Vec<i64>;
#[derive(Debug, Clone)]
pub struct Simplex {
    pub keys: Vec<LatticeKey>, pub weights: Vec<f64>
}
fn check_features(f: &Matrix) -> RegResult<()> {
    if f.nrows() == 0 || !(1..=16).contains(&f.ncols()) || !f.iter().all(|x| x.is_finite()) {
        return Err(invalid("lattice features need finite shape (n,1..16), n>0"));
    }
    Ok(())
}
/// Enclosing simplex with the scalar upstream tie-breaking convention.
pub fn enclosing_simplex(f: &[f64], with_blur: bool) -> RegResult<Simplex> {
    let d = f.len();
    if !(1..=16).contains(&d) || !f.iter().all(|x| x.is_finite()) {
        return Err(invalid("invalid lattice feature"));
    }
    let width = d + 1;
    let mut elevated = vec![0.0;
    width];
    let mut remainder = vec![0.0;
    width];
    let mut rank = vec![0_i32;
    width];
    let mut barycentric = vec![0.0;
    width + 1];
    let inv_std = (if with_blur {
        2.0_f64 / 3.0
    } else {
        1.0 / 6.0
    }).sqrt() * width as f64;
    let mut sum_feature = 0.0;
    for j in (1..=d).rev() {
        let scaled = f[j-1] * inv_std / ((j * (j+1)) as f64).sqrt();
        elevated[j] = sum_feature - j as f64 * scaled;
        sum_feature += scaled;
    }
    elevated[0] = sum_feature;
    let mut sum = 0_i64;
    for i in 0..width {
        // Check before conversion: Rust float casts otherwise silently saturate.
        if !elevated[i].is_finite() || elevated[i].abs() > (1_u64 << 46) as f64 {
            return Err(numerical("lattice coordinate overflow: rescale points or increase sigma2"));
        }
        let v = elevated[i] / width as f64;
        let upper = v.ceil() * width as f64;
        let lower = v.floor() * width as f64;
        remainder[i] = if upper-elevated[i] < elevated[i]-lower {
            upper
        } else {
            lower
        };
        sum += remainder[i] as i64;
    }
    sum /= width as i64;
    for i in 0..d {
        for j in (i+1)..width {
            if elevated[i]-remainder[i] < elevated[j]-remainder[j] {
                rank[i] += 1;
            }
            else {
                rank[j] += 1;
            }
        }
    }
    for i in 0..width {
        rank[i] += sum as i32;
        if rank[i] < 0 {
            rank[i] += width as i32;
            remainder[i] += width as f64;
        }
        else if rank[i] > d as i32 {
            rank[i] -= width as i32;
            remainder[i] -= width as f64;
        }
        if !(0..=d as i32).contains(&rank[i]) {
            return Err(numerical("invalid simplex rank"));
        }
    }
    for i in 0..width {
        let delta = (elevated[i]-remainder[i]) / width as f64;
        let a = d-rank[i] as usize;
        barycentric[a] += delta;
        barycentric[a+1] -= delta;
    }
    barycentric[0] += 1.0+barycentric[width];
    let mut keys = Vec::with_capacity(width);
    for color in 0..width {
        let key = (0..d).map(|axis| remainder[axis] as i64 + color as i64
        - if rank[axis] as usize > d-color {
            width as i64
        } else {
            0
        }).collect();
        keys.push(key);
        if !barycentric[color].is_finite() || barycentric[color] < -1e-8 {
            return Err(numerical("invalid barycentric weight"));
        }
    }
    Ok(Simplex {
        keys, weights: barycentric[..width].to_vec()
    })
}
#[derive(Debug, Clone)]
pub struct Permutohedral {
    n: usize, d: usize, with_blur: bool,
    keys: Vec<LatticeKey>, offsets: Vec<usize>, barycentric: Vec<f64>,
    neighbors: Vec<(Option<usize>, Option<usize>)>,
}
impl Permutohedral {
    pub fn new(features: &Matrix, with_blur: bool) -> RegResult<Self> {
        check_features(features)?;
        let (n,d) = features.shape();
        let capacity = n.checked_mul(d+1).ok_or_else(|| numerical("lattice size overflow"))?;
        let mut index = HashMap::<LatticeKey,usize>::with_capacity(capacity);
        let mut keys = Vec::new();
        let mut offsets = Vec::with_capacity(capacity);
        let mut barycentric = Vec::with_capacity(capacity);
        for i in 0..n {
            let f: Vec<f64> = (0..d).map(|a| features[(i,a)]).collect();
            let s = enclosing_simplex(&f,with_blur)?;
            for (key,weight) in s.keys.into_iter().zip(s.weights) {
                // Vertex order depends on insertion, not randomized HashMap iteration.
                let id = if let Some(&id) = index.get(&key) {
                    id
                } else {
                    let id = keys.len();
                    index.insert(key.clone(),id);
                    keys.push(key);
                    id
                };
                offsets.push(id);
                barycentric.push(weight);
            }
        }
        let mut neighbors = Vec::new();
        if with_blur {
            for axis in 0..=d {
                for key in &keys {
                    let mut left: Vec<i64> = key.iter().map(|x| x-1).collect();
                    let mut right: Vec<i64> = key.iter().map(|x| x+1).collect();
                    // Only d independent coordinates are stored. Never read key[d].
                    if axis < d {
                        left[axis] = key[axis]+d as i64;
                        right[axis] = key[axis]-d as i64;
                    }
                    neighbors.push((index.get(&left).copied(), index.get(&right).copied()));
                }
            }
        }
        Ok(Self {
            n,d,with_blur,keys,offsets,barycentric,neighbors
        })
    }
    pub fn lattice_size(&self) -> usize {
        self.keys.len()
    }
    /// Splat only rows start..n, then slice every feature row. Reverse reverses
    /// axis passes (the transposed linear filter), not point order.
    pub fn filter(&self, values: &Matrix, start: usize, reverse: bool) -> RegResult<Matrix> {
        if values.nrows() != self.n || values.ncols() == 0 || start > self.n
        || !values.iter().all(|x| x.is_finite()) {
            return Err(invalid("invalid lattice values or splat start"));
        }
        let count = self.keys.len();
        let channels = values.ncols();
        let mut a = Matrix::zeros(count,channels);
        let mut b = Matrix::zeros(count,channels);
        for i in start..self.n {
            for j in 0..=self.d {
                let k = i*(self.d+1)+j;
                for c in 0..channels {
                    a[(self.offsets[k],c)] += self.barycentric[k]*values[(i,c)];
                }
            }
        }
        if self.with_blur {
            for pass in 0..=self.d {
                let axis = if reverse {
                    self.d-pass
                } else {
                    pass
                };
                for i in 0..count {
                    let (left,right) = self.neighbors[axis*count+i];
                    for c in 0..channels {
                        b[(i,c)] = a[(i,c)] + left.map_or(0.0,|j| 0.5*a[(j,c)])
                        + right.map_or(0.0,|j| 0.5*a[(j,c)]);
                    }
                }
                std::mem::swap(&mut a,&mut b);
            }
        }
        // This gain is also present in probreg's no-blur mode.
        let gain = 1.0/(1.0+2.0_f64.powi(-(self.d as i32)));
        let mut out = Matrix::zeros(self.n,channels);
        for i in 0..self.n {
            for j in 0..=self.d {
                let k = i*(self.d+1)+j;
                for c in 0..channels {
                    out[(i,c)] += gain*self.barycentric[k]*a[(self.offsets[k],c)];
                }
            }
        }
        finite(&out,"lattice filter")?;
        Ok(out)
    }
}
/// Original FilterReg's observation-only, cached no-blur lattice. In contrast
/// to probreg, query vertices are not inserted and the probreg gain is absent.
#[derive(Debug, Clone)]
pub struct FixedNoBlurLattice {
    d: usize, index: HashMap<LatticeKey,usize>, splatted: Matrix,
}
impl FixedNoBlurLattice {
    pub fn new(features: &Matrix, values: &Matrix) -> RegResult<Self> {
        check_features(features)?;
        if values.nrows() != features.nrows() || values.ncols() == 0 || !values.iter().all(|x| x.is_finite()) {
            return Err(invalid("invalid no-blur values"));
        }
        let d = features.ncols();
        let mut index = HashMap::new();
        let mut offsets = Vec::new();
        let mut weights = Vec::new();
        for i in 0..features.nrows() {
            let f: Vec<f64> = (0..d).map(|a| features[(i,a)]).collect();
            let s = enclosing_simplex(&f,false)?;
            for (key,weight) in s.keys.into_iter().zip(s.weights) {
                let next_id = index.len();
                let id = *index.entry(key).or_insert(next_id);
                offsets.push(id);
                weights.push(weight);
            }
        }
        let mut splatted = Matrix::zeros(index.len(),values.ncols());
        for i in 0..features.nrows() {
            for j in 0..=d {
                let k = i*(d+1)+j;
                for c in 0..values.ncols() {
                    splatted[(offsets[k],c)] += weights[k]*values[(i,c)];
                }
            }
        }
        Ok(Self {
            d,index,splatted
        })
    }
    pub fn lattice_size(&self) -> usize {
        self.index.len()
    }
    pub fn slice(&self, queries: &Matrix) -> RegResult<Matrix> {
        check_features(queries)?;
        if queries.ncols() != self.d {
            return Err(invalid("no-blur feature dimension mismatch"));
        }
        let mut out = Matrix::zeros(queries.nrows(),self.splatted.ncols());
        for i in 0..queries.nrows() {
            let f: Vec<f64> = (0..self.d).map(|a| queries[(i,a)]).collect();
            let s = enclosing_simplex(&f,false)?;
            for (key,weight) in s.keys.iter().zip(s.weights) {
                if let Some(&j) = self.index.get(key) {
                    for c in 0..out.ncols() {
                        out[(i,c)] += weight*self.splatted[(j,c)];
                    }
                }
            }
        }
        finite(&out,"no-blur slicing")?;
        Ok(out)
    }
}
#[derive(Debug, Clone)]
pub struct FilteredValues {
    pub values: Matrix, pub vertices: usize, pub mode: String
}
pub fn lattice_transform(s: &Matrix, q: &Matrix, v: &Matrix, sigma2: f64, backend: Backend) -> RegResult<FilteredValues> {
    check_features(s)?;
    check_features(q)?;
    positive(sigma2,"sigma2")?;
    if s.ncols() != q.ncols() || s.nrows() != v.nrows() || v.ncols() == 0 || !v.iter().all(|x| x.is_finite()) {
        return Err(invalid("invalid Gaussian transform arguments"));
    }
    if backend == Backend::Direct {
        let mut out = Matrix::zeros(q.nrows(),v.ncols());
        for i in 0..q.nrows() {
            for j in 0..s.nrows() {
                let dist = (0..s.ncols()).map(|a| (q[(i,a)]-s[(j,a)]).powi(2)).sum::<f64>();
                let weight = (-0.5*dist/sigma2).exp();
                for c in 0..v.ncols() {
                    out[(i,c)] += weight*v[(j,c)];
                }
            }
        }
        finite(&out,"direct Gaussian transform")?;
        return Ok(FilteredValues {
            values:out,vertices:0,mode:"direct".into()
        });
    }
    let sigma = sigma2.sqrt();
    if backend == Backend::PermutohedralNoBlur {
        let lattice = FixedNoBlurLattice::new(&(s/sigma),v)?;
        return Ok(FilteredValues {
            values:lattice.slice(&(q/sigma))?, vertices:lattice.lattice_size(),mode:"original_noblur".into()
        });
    }
    let features = Matrix::from_fn(q.nrows()+s.nrows(),s.ncols(),|i,a| {
        if i < q.nrows() {
            q[(i,a)]/sigma
        } else {
            s[(i-q.nrows(),a)]/sigma
        }
    });
    let values = Matrix::from_fn(features.nrows(),v.ncols(),|i,c| {
        if i < q.nrows() {
            0.0
        } else {
            v[(i-q.nrows(),c)]
        }
    });
    let lattice = Permutohedral::new(&features,true)?;
    let (lattice,mode) = if backend == Backend::Probreg && lattice.lattice_size() as f64 > 0.015*s.nrows() as f64 {
        (Permutohedral::new(&features,false)?,"probreg_noblur")
    } else {
        (lattice,"blur")
    };
    let filtered = lattice.filter(&values,q.nrows(),false)?;
    Ok(FilteredValues {
        values:filtered.rows(0,q.nrows()).into_owned(),vertices:lattice.lattice_size(),mode:mode.into()
    })
}
