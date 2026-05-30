use criterion::{criterion_group, criterion_main, Criterion};

fn scoring_benchmarks(_c: &mut Criterion) {
    // Placeholder — will benchmark score_hamming, score_int3, score_binary_rerank
    // once data loading is tested.
}

criterion_group!(benches, scoring_benchmarks);
criterion_main!(benches);
