use plonky2_field::fft::fft;
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::polynomial::PolynomialCoeffs;
use plonky2_field::types::Field;
use rand::random;

fn random_fr() -> u64 {
    let fr: u64 = random();
    fr % 0xffffffff00000001
}

fn main() {
    let domain_size = 1usize << 10;

    let v: Vec<u64> = (0..domain_size).map(|_| random_fr()).collect();
    let buffer = v.clone();

    let coeffs = buffer
        .iter()
        .map(|i| GoldilocksField::from_canonical_u64(*i))
        .collect::<Vec<GoldilocksField>>();
    let coefficients = PolynomialCoeffs { coeffs };
    let points = fft(coefficients.clone());
    println!("result: {:?}", points);
}
