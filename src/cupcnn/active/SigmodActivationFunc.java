package cupcnn.active;

public class SigmodActivationFunc extends ActivationFunc {
	public static final String TYPE = "SigmodActivationFunc";
	
	@Override
	public double active(double in) {
		// TODO Auto-generated method stub
		double result = 0;
		result = 1.0/(1.0+Math.exp(-in));
		return result;
	}

	@Override
	public double diffActive(double in) {
		// TODO Auto-generated method stub
		double result = 0.0;
		result = (Math.exp(-in))/((1+Math.exp(-in))*(1+Math.exp(-in)));
		return result;
	}
	
	@Override
	public String getType(){
		return TYPE;
	}
}
