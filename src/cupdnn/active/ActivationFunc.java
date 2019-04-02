package cupdnn.active;

public abstract class ActivationFunc {
	public abstract float active(float in);
	public abstract float diffActive(float in);
	public abstract String getType();
}
