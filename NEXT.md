# Next Steps for improved_tunnel

*Updated based on 2024-2025 research trends in quantum tunneling calculations*

---

## ✅ Completed

### Unit Tests (151 tests passing)
```
tests/
├── test_molecule.py       # 21 tests - Atom, Molecule, geometry, isotopes
├── test_reduced_mass.py   # 12 tests - Torsional reduced mass calculations
├── test_interpolation.py  # 14 tests - Periodic cubic spline
├── test_tunneling.py      # 18 tests - WKB, Eckart, SCT methods
├── test_kinetics.py       # 15 tests - Rate calculations, Arrhenius
├── test_workflow.py       # 20 tests - Full workflow integration
├── test_ipi.py            # 24 tests - i-PI socket communication
└── test_visualization.py  # 19 tests - Plotting functions
```

### i-PI Interface Module
Created `instanton/` module for Ring-Polymer Instanton calculations:
- `ipi_client.py` - Socket client for i-PI communication (POSDATA/GETFORCE protocol)
- `ipi_driver.py` - Psi4 driver that acts as i-PI client (receives geometries, returns forces)
- `ipi_input.py` - i-PI XML input file generator for various calculation types:
  - `generate_instanton_input()` - Ring-polymer instanton optimization
  - `generate_geop_input()` - Geometry optimization
  - `generate_neb_input()` - Nudged elastic band calculations
  - `generate_md_input()` - Molecular dynamics
  - `create_driver_script()` - Generate Psi4 driver scripts
- Mock driver for testing without Psi4
- Full test coverage with mock i-PI server

### Visualization Module
Created `visualization/` module:
- `pes_plot.py` - PES plotting (single, comparison, with tunneling region)
- `transmission_plot.py` - Transmission coefficient T(E) plots
- `arrhenius_plot.py` - Arrhenius plots, KIE analysis
- `instanton_plot.py` - Ring-polymer visualization, instanton path plots

### Benchmark Infrastructure
Created `benchmarks/` module:
- `reference_data.py` - Literature values for malonaldehyde (21.6 cm⁻¹), H₂O₂, formic acid dimer, tropolone
- `molecules.py` - Factory functions for benchmark molecules with accurate geometries
- `run_benchmark.py` - Command-line benchmark runner

---

## Tier 1: Fundamental Improvements (High Priority)

### 1. ✅ i-PI Integration Complete
The i-PI interface is implemented and tested. Next: test with actual i-PI installation.

**To run with real i-PI:**
```bash
# Start i-PI server
i-pi input.xml &

# Run Psi4 driver
python -c "
from improved_tunnel.instanton import Psi4IPIDriver
driver = Psi4IPIDriver(port=31415, method='MP2', basis='cc-pVDZ')
driver.run(symbols=['H', 'O', 'O', 'H'])
"
```

### 2. Ring-Polymer Instanton Optimization
Now that the i-PI interface exists, implement the instanton-specific workflow:
- Generate initial ring-polymer path between reactant/product
- Set up i-PI input file for instanton optimization
- Post-process instanton for rate calculation

### 3. Benchmark with Real Calculations
Run malonaldehyde and H₂O₂ benchmarks with actual Psi4:
```bash
python -m improved_tunnel.benchmarks.run_benchmark --system malonaldehyde --real
```

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
✅ DONE:   Unit tests (151 passing)
✅ DONE:   i-PI interface module (client, driver, input generator)
✅ DONE:   Visualization module (PES, transmission, Arrhenius, instanton)
✅ DONE:   Benchmark infrastructure (malonaldehyde, H₂O₂)

NEXT:     Test with real i-PI installation
NEXT:     Run benchmarks with real Psi4
THEN:     ORCA integration
LATER:    Delta-ML, GPR acceleration
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
