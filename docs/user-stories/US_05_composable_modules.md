# US-05: Compose Modules with Type-Checked Shapes

**As** a researcher,
**I want** to compose two BWSK-defined modules (`encoder >> decoder`) and have the type system verify shape compatibility at definition time,
**So that** shape errors are caught before training begins.

## Acceptance Criteria

- [ ] `>>` operator composes BWSK modules
- [ ] Shape mismatch raises error at composition time
- [ ] Composed module is a valid `nn.Module`

## Spec Reference

- [SPEC-004: Architecture DSL](../specs/SPEC_004_architecture_dsl.md)
