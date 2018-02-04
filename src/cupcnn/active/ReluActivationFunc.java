package cupcnn.active;

public class ReluActivationFunc extends ActivationFunc {
	public static final String TYPE = "ReluActivationFunc";
	
	@Override
	public double active(double in) {
		// TODO Auto-generated method stub
		double result = in > 0 ? in:0;
		return result;
	}

	@Override
	public double diffActive(double in) {
		// TODO Auto-generated method stub
		double result = in<=0 ? 0.1:1.0;
		return result;
	}
	
	@Override
	public String getType(){
		return TYPE;
	}

}
