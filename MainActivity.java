package trial.myapplication;


import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.apache.commons.math3.stat.StatUtils;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;


public class MainActivity extends AppCompatActivity {

    private double[] inputArray;
    private Interpreter tflite;
    private static final int NUM_CLASSES = 2;
    private static final String[] CLASS_LABELS = {"True", "False"};
    private String[] runPrediction;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        final TextView textView1 = findViewById(R.id.textView1);
        Button button1 = findViewById(R.id.button1);

        final TextView textView2 = findViewById(R.id.textView2);
        Button button2 = findViewById(R.id.button2);

        final TextView result = findViewById(R.id.result);
        Button predict = findViewById(R.id.predict);

        button1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String str = "";
                try {
                    InputStream inputStream = getAssets().open("EO_data.txt");
                    int size = inputStream.available();
                    byte[] buffer = new byte[size];
                    inputStream.read(buffer);

                    str = new String(buffer);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                String[] stringArray = str.split(",");
                inputArray = new double[stringArray.length];
                for (int i = 0; i < stringArray.length; i++) {
                    inputArray[i] = Double.parseDouble(stringArray[i]);
                }
                String inputArrayString = Arrays.toString(inputArray);
                textView1.setText(inputArrayString);

            }
        });

        button2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String str = "";
                try {
                    InputStream inputStream = getAssets().open("EC_data.txt");
                    int size = inputStream.available();
                    byte[] buffer = new byte[size];
                    inputStream.read(buffer);

                    str = new String(buffer);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                String[] stringArray = str.split(",");
                inputArray = new double[stringArray.length];
                for (int i = 0; i < stringArray.length; i++) {
                    inputArray[i] = Double.parseDouble(stringArray[i]);
                }
                String inputArrayString = Arrays.toString(inputArray);
                textView2.setText(inputArrayString);
            }
        });

        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String predictedClassLabel = MainActivity.this.runPrediction(inputArray);
                result.setText("Predicted class:" + predictedClassLabel);
            }
        }
        );
    }

    public String runPrediction(double[] inputData) {

        double[] normalizedArray = StatUtils.normalize(inputData);
        float[] floatArray = new float[normalizedArray.length];
        for (int i = 0; i< normalizedArray.length; i++) {
            floatArray[i] = (float) normalizedArray[i];
        }

        TensorBuffer inputBuffer = TensorBuffer.createFixedSize(new int[]{1, floatArray.length}, DataType.FLOAT32);
        inputBuffer.loadArray(floatArray);

        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(new int[]{1, NUM_CLASSES}, DataType.FLOAT32);
        tflite.run(inputBuffer.getBuffer(), outputBuffer.getBuffer());

        float[] probabilities = outputBuffer.getFloatArray();
        int maxIndex = 0;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] < probabilities[maxIndex]) {
                maxIndex = i;
            }
        }
        String predictedClassLabel = CLASS_LABELS[maxIndex];
        return predictedClassLabel;
    }

    private ByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = getAssets().openFd("Method1.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }

}
