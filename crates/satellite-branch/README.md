# satellite-branch

Git-style branch model for parallel constraint solving.

## Features

- **Constraint branching** - Fork solver state on constraint splits
- **Reference counting** - Automatic branch cleanup
- **Failure propagation** - Child failure propagates to parent
- **Lock-free status tracking** via skiplist

## Concepts

Unlike traditional CDCL decision trees, Satellite uses semantic constraint branches:

```
Parent Branch (refcount=3)
    ├─ Child A: constraint variant 1
    ├─ Child B: constraint variant 2
    └─ Child C: constraint variant 3
```

## License

MIT
