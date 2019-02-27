package cupdnn.active;

public class SigmodActivationFunc extends ActivationFunc {
	public static final String TYPE = "SigmodActivationFunc";
	
	@Override
	public float active(float in) {
		// TODO Auto-generated method stub
		float result = 0;
		result = 1.0f/(1.0f+(float)Math.exp(-in));
		return result;
	}

	@Override
	public float diffActive(float in) {
		// TODO Auto-generated method stub
		float result = 0.0f;
		result = (float) ((Math.exp(-in))/((1+Math.exp(-in))*(1+Math.exp(-in))));
		return result;
	}
	
	@Override
	public String getType(){
		return TYPE;
	}
}
