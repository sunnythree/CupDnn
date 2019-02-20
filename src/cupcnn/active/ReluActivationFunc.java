package cupcnn.active;

public class ReluActivationFunc extends ActivationFunc {
	public static final String TYPE = "ReluActivationFunc";
	
	@Override
	public float active(float in) {
		// TODO Auto-generated method stub
		float result = in > 0 ? in:0;
		return result;
	}

	@Override
	public float diffActive(float in) {
		// TODO Auto-generated method stub
		float result = in<=0 ? 0.01f:1.0f;
		return result;
	}
	
	@Override
	public String getType(){
		return TYPE;
	}

}
