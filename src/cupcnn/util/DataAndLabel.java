package cupcnn.util;

public class DataAndLabel {
	public float[] label;
	public float[] data;
	public DataAndLabel(int dataSize,int labelSize) {
		data = new float[dataSize];
		label = new float[labelSize];
	}
	public void setData(float[] d,float[] l) {
		System.arraycopy(d, 0, data, 0, d.length);
		System.arraycopy(l, 0, label, 0, l.length);
	}
}
