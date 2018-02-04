package cupcnn.active;

public class TanhActivationFunc extends ActivationFunc{
	public static final String TYPE = "TanhActivationFunc";

	private double tanh(double in){
		double ef = Math.exp(in);
		double efx = Math.exp(-in);
		return (ef-efx)/(ef+efx);
	}
	@Override
	public double active(double in) {
		// TODO Auto-generated method stub
		return tanh(in);
	}

	@Override
	public double diffActive(double in) {
		// TODO Auto-generated method stub
		return (1-tanh(in)*tanh(in));
	}
	
	@Override
	public String getType(){
		return TYPE;
	}

}
