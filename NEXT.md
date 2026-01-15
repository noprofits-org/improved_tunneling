# Next Steps for improved_tunnel

*Updated based on 2024-2025 research trends in quantum tunneling calculations*

---

## Tier 1: Fundamental Improvements (High Priority)

### 1. Implement Ring-Polymer Instanton (RPI) Method
**Why**: RPI is the current gold standard for tunneling calculations, not WKB/SCT.

**Options**:
- **Interface with i-PI 3.0** (recommended) - Released August 2024, well-tested implementation
- Implement from scratch using Richardson's formulation

```python
# Example architecture
class RingPolymerInstanton:
    def __init__(self, n_beads=32, temperature=300):
        self.n_beads = n_beads
        self.beta_n = 1 / (n_beads * kB * temperature)

    def optimize_instanton(self, initial_path, calculator):
        """Find saddle point on ring-polymer potential surface."""
        pass

    def compute_tunneling_rate(self, instanton, hessian):
        """Rate from instanton action and fluctuation determinant."""
        pass
```

**Key reference**: Litman et al., [J. Chem. Phys. 159, 014111 (2023)](https://pubs.aip.org/aip/jcp/article/159/1/014111) - Perturbatively corrected RPI achieves 2% accuracy for malonaldehyde.

### 2. Add Unit Tests
Essential for reliability, especially before adding instanton:
```
tests/
├── test_molecule.py
├── test_reduced_mass.py
├── test_interpolation.py
├── test_wkb.py           # Compare with analytical Eckart
├── test_rates.py
└── test_integration.py   # Full workflow
```

### 3. Benchmark Against Literature
Validate against established systems:

| System | Experimental | Target Accuracy |
|--------|--------------|-----------------|
| Malonaldehyde | 21.6 cm⁻¹ | ±2 cm⁻¹ |
| H₂O₂ barrier | ~7 kcal/mol | ±0.5 kcal/mol |
| H₂O₂ dihedral | 111.5° | ±1° |

**Key references**:
- Koput, J. J. Mol. Spectrosc. 1986, 115, 438.
- Pelz, G. et al. J. Mol. Spectrosc. 1993, 159, 507.

---

## Tier 2: Accuracy Enhancement (Medium-High Priority)

### 4. Delta-ML Integration
Standard practice for CCSD(T)-quality results at MP2 cost:

1. Train on MP2/cc-pVTZ PES along reaction coordinate
2. Select 25-50 points along instanton path
3. Compute CCSD(T)-F12 corrections at selected points
4. Train GPR or polynomial correction model

**Reference**: Käser et al., [JCTC 2023](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00790)

### 5. Multi-Code Support (ORCA, PySCF)
Add interfaces beyond Psi4:

| Package | Why | Priority |
|---------|-----|----------|
| **ORCA** | Free academic; excellent CCSD(T) | High |
| **PySCF** | Pure Python; GPU acceleration | High |
| **xTB** | Fast semiempirical screening | Medium |

Consider using ASE calculator interface for unified API:
```python
from ase.calculators.orca import ORCA
calc = ORCA(method='CCSD(T)', basis='cc-pVTZ')
```

### 6. Visualization Module
Create `improved_tunnel/visualization/`:
- `plot_pes()` - Energy vs dihedral
- `plot_transmission()` - T(E) curves
- `plot_arrhenius()` - ln(k) vs 1/T
- `plot_instanton_path()` - Ring-polymer bead positions

---

## Tier 3: Expansion (Medium Priority)

### 7. Additional Benchmark Systems
Extend beyond H₂O₂:

**Proton transfer** (dramatic tunneling):
- Malonaldehyde - THE benchmark, KIE ~10-20
- Tropolone - 15 atoms, tests scalability
- Formic acid dimer - double proton transfer

**Heavy atom tunneling**:
- Cyclopropane ring opening
- Carbene rearrangements

### 8. GPR Acceleration for Instanton
Gaussian Process Regression to reduce QC calls by 10x:
- Adaptive sampling along tunneling path
- Uncertainty-guided point selection

**Reference**: Zaverkin et al., [JCTC 2024](https://pubs.acs.org/doi/full/10.1021/acs.jctc.4c00158)

### 9. Command-Line Interface
```bash
improved-tunnel run h2o2.xyz --method MP2 --basis cc-pVDZ
improved-tunnel instanton h2o2.xyz --n-beads 32 --temperature 300
improved-tunnel benchmark malonaldehyde --compare-experiment
```

---

## Tier 4: Reconsider/Deprecate

### Eckart Refinements (Lower Priority)
The current Eckart implementation is useful for quick estimates but:
- 1D approximation struggles with multidimensional effects
- Instanton supersedes it for quantitative work
- Consider keeping as "screening" method only

### WKB Improvements (Lower Priority)
WKB typically overestimates tunneling half-lives:
- Useful for initial screening
- Not competitive with RPI for quantitative predictions
- Keep current implementation, don't invest heavily

---

## Key Literature

1. **i-PI 3.0**: Litman et al., [J. Chem. Phys. 161, 062504 (2024)](https://pubs.aip.org/aip/jcp/article/161/6/062504)
2. **Perturbatively corrected RPI**: [J. Chem. Phys. 159, 014111 (2023)](https://pubs.aip.org/aip/jcp/article/159/1/014111)
3. **Delta-ML for tunneling**: [JCTC 2023](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00790)
4. **Tropolone benchmark**: [JACS 2023](https://pubs.acs.org/doi/10.1021/jacs.3c00769)
5. **GPR for instanton**: [JCTC 2024](https://pubs.acs.org/doi/full/10.1021/acs.jctc.4c00158)

---

## Recommended Execution Order

```
Week 1-2:  Unit tests for existing code
Week 3-4:  i-PI interface OR basic instanton implementation
Week 5:    Malonaldehyde benchmark
Week 6-7:  ORCA integration
Week 8:    Visualization module
Week 9+:   Delta-ML, GPR acceleration
```

---

## Summary

Your H₂O₂ focus is excellent - it's a fundamental benchmark system. The main shift from the original roadmap:

| Original Priority | New Priority |
|-------------------|--------------|
| Higher-level theory | → Delta-ML approach |
| Fix Eckart | → Implement Instanton (supersedes Eckart) |
| CLI | → Multi-code support first |

The **instanton method** should be the core focus - it's what differentiates a research-grade tunneling package from a pedagogical one.
