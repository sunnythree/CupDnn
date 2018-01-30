package cupcnn.active;

public class ReluActivationFunc extends ActivationFunc {

	@Override
	public double active(double in) {
		// TODO Auto-generated method stub
		double result = in > 0 ? 0.1*in:0;
		return result;
	}

	@Override
	public double diffActive(double in) {
		// TODO Auto-generated method stub
		double result = in<=0 ? 0.001:0.1;
		return result;
	}

}
