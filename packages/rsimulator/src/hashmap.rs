use rustc_hash::FxHasher;
/// Centralised HashMap typedefs to allow swapping implementation easily.
///
/// This file defines `Map` and `Set` aliases using `FxHashMap`/`FxHashSet`
/// (fast, deterministic hashing). Other modules should import `crate::hashmap::*`
/// and use `Map` / `Set` instead of `std::collections::HashMap` / `HashSet`.
use std::collections::{HashMap as StdHashMap, HashSet as StdHashSet};
use std::hash::BuildHasherDefault;

/// Deterministic, fast hash map / set using FxHasher as the build hasher.
pub type Map<K, V> = StdHashMap<K, V, BuildHasherDefault<FxHasher>>;
pub type Set<K> = StdHashSet<K, BuildHasherDefault<FxHasher>>;
