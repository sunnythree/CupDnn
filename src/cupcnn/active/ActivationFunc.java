package cupcnn.active;

public abstract class ActivationFunc {
	public abstract double active(double in);
	public abstract double diffActive(double in);
	public abstract String getType();
}
