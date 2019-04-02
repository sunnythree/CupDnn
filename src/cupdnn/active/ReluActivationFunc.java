package cupdnn.active;

public class ReluActivationFunc extends ActivationFunc {
	public static final String TYPE = "ReluActivationFunc";
	
	@Override
	public float active(float in) {
		// TODO Auto-generated method stub
		return Math.max(0, in);
	}

	@Override
	public float diffActive(float in) {
		// TODO Auto-generated method stub
		float result = in<=0 ? 0.0f:1.0f;
		return result;
	}
	
	@Override
	public String getType(){
		return TYPE;
	}

}
