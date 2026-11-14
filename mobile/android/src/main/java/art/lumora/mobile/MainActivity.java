package art.lumora.mobile;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class MainActivity extends Activity {
    static {
        System.loadLibrary("lumora_ffi");
        System.loadLibrary("lumora_jni");
    }

    // Native JNI methods
    public native long nativeInitEngine(Bitmap bitmap);
    public native void nativeDestroyEngine(long handle);
    public native int nativeApplyExposure(long handle, float ev);
    public native int nativeApplyContrast(long handle, float factor);
    public native int nativeApplySaturation(long handle, float factor);
    public native int nativeAddUPoint(long handle, float x, float y, float radius, float brightness, float contrast, float saturation);
    public native int nativeClearUPoints(long handle);
    public native int nativeApplyAiEnhancement(long handle, int opType, float strength);
    public native int nativeRenderPreview(long handle, Bitmap outBitmap, int maxW, int maxH);
    public native byte[] nativeExportImage(long handle, int format, int quality, int tileSize);

    private long mEngineHandle = 0;
    private Bitmap mOriginalBitmap;
    private Bitmap mPreviewBitmap;
    private ImageView mCanvasView;
    private TextView mStatusText;
    private LinearLayout mControlsLayout;

    private float mExposure = 0.0f;
    private float mContrast = 1.0f;
    private float mSaturation = 1.0f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xFF121212); // Pro dark background

        // 1. Top Header Bar
        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.HORIZONTAL);
        header.setPadding(32, 48, 32, 24);
        header.setBackgroundColor(0xFF1E1E1E);

        TextView title = new TextView(this);
        title.setText("LUMORA STUDIO");
        title.setTextColor(0xFFFFFFFF);
        title.setTextSize(20);
        title.setTypeface(null, android.graphics.Typeface.BOLD);
        header.addView(title);

        View spacer = new View(this);
        LinearLayout.LayoutParams spacerParams = new LinearLayout.LayoutParams(0, 1, 1.0f);
        header.addView(spacer, spacerParams);

        Button pickBtn = new Button(this);
        pickBtn.setText("OPEN PHOTO");
        pickBtn.setTextColor(0xFFFFFFFF);
        pickBtn.setBackgroundColor(0xFF333333);
        pickBtn.setOnClickListener(v -> openGallery());
        header.addView(pickBtn);

        Button exportBtn = new Button(this);
        exportBtn.setText("EXPORT JPEG");
        exportBtn.setTextColor(0xFF121212);
        exportBtn.setBackgroundColor(0xFF00E676); // Accent green
        exportBtn.setOnClickListener(v -> exportFullResolution());
        header.addView(exportBtn);

        root.addView(header);

        // 2. Status Label
        mStatusText = new TextView(this);
        mStatusText.setText("Initializing Lumora Rust DAG Engine...");
        mStatusText.setTextColor(0xFF888888);
        mStatusText.setPadding(32, 12, 32, 12);
        mStatusText.setTextSize(12);
        root.addView(mStatusText);

        // 3. Central Canvas
        FrameLayout canvasFrame = new FrameLayout(this);
        LinearLayout.LayoutParams frameParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1.0f);
        canvasFrame.setLayoutParams(frameParams);

        mCanvasView = new ImageView(this);
        mCanvasView.setScaleType(ImageView.ScaleType.FIT_CENTER);
        FrameLayout.LayoutParams imgParams = new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT);
        canvasFrame.addView(mCanvasView, imgParams);
        root.addView(canvasFrame);

        // 4. Tab Navigation (Develop, Selective, AI Lab, Presets)
        HorizontalScrollView tabScroll = new HorizontalScrollView(this);
        tabScroll.setBackgroundColor(0xFF1A1A1A);
        LinearLayout tabRow = new LinearLayout(this);
        tabRow.setOrientation(LinearLayout.HORIZONTAL);
        tabRow.setPadding(16, 8, 16, 8);

        String[] tabs = {"DEVELOP (Lightroom)", "SELECTIVE (Snapseed)", "AI ENHANCE (Models)", "PRESETS"};
        for (int i = 0; i < tabs.length; i++) {
            final int tabIdx = i;
            Button tabBtn = new Button(this);
            tabBtn.setText(tabs[i]);
            tabBtn.setTextColor(i == 0 ? 0xFF00E676 : 0xFFCCCCCC);
            tabBtn.setBackgroundColor(0xFF242424);
            tabBtn.setPadding(32, 16, 32, 16);
            tabBtn.setOnClickListener(v -> showTab(tabIdx));
            tabRow.addView(tabBtn);
        }
        tabScroll.addView(tabRow);
        root.addView(tabScroll);

        // 5. Controls Drawer
        mControlsLayout = new LinearLayout(this);
        mControlsLayout.setOrientation(LinearLayout.VERTICAL);
        mControlsLayout.setPadding(32, 24, 32, 32);
        mControlsLayout.setBackgroundColor(0xFF181818);
        root.addView(mControlsLayout);

        setContentView(root);

        // Generate colorful sample test pattern and init engine
        createSampleImage();
        showTab(0);
    }

    private void createSampleImage() {
        int w = 1080;
        int h = 1440;
        mOriginalBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(mOriginalBitmap);

        Paint paint = new Paint();
        // Background gradient
        for (int y = 0; y < h; y++) {
            float ratio = (float) y / h;
            int r = (int) (40 + ratio * 180);
            int g = (int) (60 + ratio * 120);
            int b = (int) (120 - ratio * 40);
            paint.setColor(Color.rgb(r, g, b));
            canvas.drawLine(0, y, w, y, paint);
        }

        // Color blocks for testing HSL and U-Point
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(0xFFFF5252); // Red
        canvas.drawCircle(300, 500, 180, paint);

        paint.setColor(0xFF448AFF); // Blue
        canvas.drawCircle(780, 500, 180, paint);

        paint.setColor(0xFFFFD740); // Yellow/Amber
        canvas.drawCircle(540, 850, 220, paint);

        paint.setColor(0xFF69F0AE); // Green
        canvas.drawRect(200, 1150, 880, 1350, paint);

        paint.setColor(0xFFFFFFFF);
        paint.setTextSize(48);
        paint.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        canvas.drawText("LUMORA ENGINE - 1080x1440", 180, 250, paint);

        initEngineWithBitmap(mOriginalBitmap);
    }

    private void initEngineWithBitmap(Bitmap bmp) {
        if (mEngineHandle != 0) {
            nativeDestroyEngine(mEngineHandle);
            mEngineHandle = 0;
        }

        mOriginalBitmap = bmp;
        long start = System.currentTimeMillis();
        mEngineHandle = nativeInitEngine(bmp);
        long ms = System.currentTimeMillis() - start;

        mPreviewBitmap = Bitmap.createBitmap(bmp.getWidth(), bmp.getHeight(), Bitmap.Config.ARGB_8888);
        renderPreview();
        mStatusText.setText("Engine ready in " + ms + "ms | Resolution: " + bmp.getWidth() + "x" + bmp.getHeight());
    }

    private void renderPreview() {
        if (mEngineHandle == 0 || mPreviewBitmap == null) return;

        long start = System.currentTimeMillis();
        int res = nativeRenderPreview(mEngineHandle, mPreviewBitmap, mPreviewBitmap.getWidth(), mPreviewBitmap.getHeight());
        long ms = System.currentTimeMillis() - start;

        if (res == 0) {
            mCanvasView.setImageBitmap(mPreviewBitmap);
            mStatusText.setText("Rust DAG Render: " + ms + "ms | Exp: " + String.format("%.2f", mExposure) + " | Sat: " + String.format("%.2f", mSaturation));
        } else {
            mStatusText.setText("Render error: " + res);
        }
    }

    private void showTab(int tabIndex) {
        mControlsLayout.removeAllViews();

        if (tabIndex == 0) {
            // Develop Tab
            addSlider("Exposure (-3 to +3 EV)", -30, 30, (int) (mExposure * 10), val -> {
                mExposure = val / 10.0f;
                nativeApplyExposure(mEngineHandle, mExposure);
                renderPreview();
            });

            addSlider("Contrast (0.5x to 2.0x)", 5, 20, (int) (mContrast * 10), val -> {
                mContrast = val / 10.0f;
                nativeApplyContrast(mEngineHandle, mContrast);
                renderPreview();
            });

            addSlider("Saturation (0.0x to 2.0x)", 0, 20, (int) (mSaturation * 10), val -> {
                mSaturation = val / 10.0f;
                nativeApplySaturation(mEngineHandle, mSaturation);
                renderPreview();
            });
        } else if (tabIndex == 1) {
            // Selective Tab (Snapseed U-Point)
            TextView label = new TextView(this);
            label.setText("Snapseed U-Point Selective Control Points (Lab Space)");
            label.setTextColor(0xFF00E676);
            label.setPadding(0, 0, 0, 16);
            mControlsLayout.addView(label);

            Button addPinBtn = new Button(this);
            addPinBtn.setText("+ Add U-Point at Center (540, 850)");
            addPinBtn.setBackgroundColor(0xFF2E7D32);
            addPinBtn.setTextColor(0xFFFFFFFF);
            addPinBtn.setOnClickListener(v -> {
                // Add U-Point centered on yellow circle with radius 250px and +0.6 brightness boost
                nativeAddUPoint(mEngineHandle, 540, 850, 250, 0.6f, 0.2f, 0.4f);
                renderPreview();
                Toast.makeText(this, "U-Point selective mask applied!", Toast.LENGTH_SHORT).show();
            });
            mControlsLayout.addView(addPinBtn);

            Button clearBtn = new Button(this);
            clearBtn.setText("Clear All U-Points");
            clearBtn.setBackgroundColor(0xFF424242);
            clearBtn.setTextColor(0xFFFFFFFF);
            clearBtn.setOnClickListener(v -> {
                nativeClearUPoints(mEngineHandle);
                renderPreview();
            });
            mControlsLayout.addView(clearBtn);
        } else if (tabIndex == 2) {
            // AI Lab Tab
            TextView label = new TextView(this);
            label.setText("On-Device AI Models (Local Neural Inference)");
            label.setTextColor(0xFF00E676);
            label.setPadding(0, 0, 0, 16);
            mControlsLayout.addView(label);

            String[] aiOps = {"DnCNN Denoising", "MiDaS Depth Bokeh", "Real-ESRGAN 4x Upscale", "Zero-DCE Low Light Boost"};
            int[] opTypes = {0, 1, 3, 4};

            for (int i = 0; i < aiOps.length; i++) {
                final int op = opTypes[i];
                final String name = aiOps[i];
                Button btn = new Button(this);
                btn.setText("Apply " + name);
                btn.setBackgroundColor(0xFF311B92);
                btn.setTextColor(0xFFFFFFFF);
                btn.setOnClickListener(v -> {
                    long t0 = System.currentTimeMillis();
                    nativeApplyAiEnhancement(mEngineHandle, op, 0.7f);
                    long duration = System.currentTimeMillis() - t0;
                    renderPreview();
                    Toast.makeText(this, name + " executed locally in " + duration + "ms", Toast.LENGTH_SHORT).show();
                });
                mControlsLayout.addView(btn);
            }
        } else if (tabIndex == 3) {
            // Presets Tab
            TextView label = new TextView(this);
            label.setText("One-Tap Color Presets (.lumora & .cube LUTs)");
            label.setTextColor(0xFF00E676);
            label.setPadding(0, 0, 0, 16);
            mControlsLayout.addView(label);

            String[] presets = {"Cinematic Teal & Orange", "Moody Vintage Film", "High Contrast Monochrome"};
            for (String p : presets) {
                Button btn = new Button(this);
                btn.setText("Load Preset: " + p);
                btn.setBackgroundColor(0xFF37474F);
                btn.setTextColor(0xFFFFFFFF);
                btn.setOnClickListener(v -> {
                    // Apply preset adjustment combo
                    nativeApplyExposure(mEngineHandle, 0.2f);
                    nativeApplyContrast(mEngineHandle, 1.3f);
                    nativeApplySaturation(mEngineHandle, 1.15f);
                    renderPreview();
                    Toast.makeText(this, "Preset applied: " + p, Toast.LENGTH_SHORT).show();
                });
                mControlsLayout.addView(btn);
            }
        }
    }

    interface ValueCallback {
        void onValueChanged(int val);
    }

    private void addSlider(String title, int min, int max, int current, ValueCallback cb) {
        TextView label = new TextView(this);
        label.setText(title + ": " + (current / 10.0f));
        label.setTextColor(0xFFE0E0E0);
        label.setPadding(0, 8, 0, 4);
        mControlsLayout.addView(label);

        SeekBar seekBar = new SeekBar(this);
        seekBar.setMax(max - min);
        seekBar.setProgress(current - min);
        seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                int actual = progress + min;
                label.setText(title + ": " + (actual / 10.0f));
                cb.onValueChanged(actual);
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        mControlsLayout.addView(seekBar);
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(intent, 101);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == 101 && resultCode == RESULT_OK && data != null && data.getData() != null) {
            try {
                Uri uri = data.getData();
                InputStream is = getContentResolver().openInputStream(uri);
                Bitmap bmp = BitmapFactory.decodeStream(is);
                if (bmp != null) {
                    initEngineWithBitmap(bmp);
                    Toast.makeText(this, "Photo loaded: " + bmp.getWidth() + "x" + bmp.getHeight(), Toast.LENGTH_SHORT).show();
                }
            } catch (Exception e) {
                Toast.makeText(this, "Failed to load photo: " + e.getMessage(), Toast.LENGTH_LONG).show();
            }
        }
    }

    private void exportFullResolution() {
        if (mEngineHandle == 0) return;

        mStatusText.setText("Exporting full-resolution 12MP-48MP tiled image...");
        long start = System.currentTimeMillis();
        // format 0 = JPEG, quality 92, tile_size 512
        byte[] jpegBytes = nativeExportImage(mEngineHandle, 0, 92, 512);
        long duration = System.currentTimeMillis() - start;

        if (jpegBytes != null && jpegBytes.length > 0) {
            try {
                File dir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
                if (dir == null) dir = getFilesDir();
                File outFile = new File(dir, "Lumora_Export_" + System.currentTimeMillis() + ".jpg");
                FileOutputStream fos = new FileOutputStream(outFile);
                fos.write(jpegBytes);
                fos.close();

                mStatusText.setText("Exported in " + duration + "ms! (" + (jpegBytes.length / 1024) + " KB) -> " + outFile.getName());
                Toast.makeText(this, "Saved to " + outFile.getAbsolutePath(), Toast.LENGTH_LONG).show();
            } catch (Exception e) {
                Toast.makeText(this, "Export save error: " + e.getMessage(), Toast.LENGTH_LONG).show();
            }
        } else {
            Toast.makeText(this, "Export failed in Rust engine", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    protected void onDestroy() {
        if (mEngineHandle != 0) {
            nativeDestroyEngine(mEngineHandle);
            mEngineHandle = 0;
        }
        super.onDestroy();
    }
}
