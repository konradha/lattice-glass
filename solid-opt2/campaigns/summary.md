# solid-opt2 campaign summary

Protocol: L21, 8 independent chains per beta, beta=linspace(0.01, 2.0, 8),
nonlocal and local modes, CPU and Metal.

| Backend | wall |
|---|---:|
| CPU | 149.6 s |
| Metal | 145.6 s |

Nonlocal campaigns are bitwise equal between CPU and Metal.
Metal L21 local uses the 21-colour schedule; its CPU/Metal comparison is the
statistical gate.

Gates:
- exact colored schedule: PASS
- statistical both-mode: PASS, joint p=0.692
- high-beta nonlocal: PASS, CPU 88.3 s, Metal 74.8 s
