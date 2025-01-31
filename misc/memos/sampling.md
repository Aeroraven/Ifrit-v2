### Rendering Equation
$$
L_o(\mathbf{x}, \omega_o) = L_e(\mathbf{x}, \omega_o) + \int_{\Omega} f(\mathbf{x}, \omega_i, \omega_o) L_i(\mathbf{x}, \omega_i) (\omega_i \cdot \mathbf{n}) d\omega_i
$$

### Monte Carlo Integration
$$
\int f(x) dx \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i)
,
\text{where } x_i \text{ is sampled from } p(x)
$$


### Importance Sampling
$$

\int f(x) dx = \int \frac{f(x)}{p(x)} p(x) dx \approx \frac{1}{N} \sum_{i=1}^{N} \frac{f(x_i)}{p(x_i)}

$$