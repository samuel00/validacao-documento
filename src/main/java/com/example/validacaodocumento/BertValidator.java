package com.example.validacaodocumento;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.huggingface.tokenizers.Encoding;

import ai.onnxruntime.*;

import java.nio.LongBuffer;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;

public class BertValidator {
    private static final String MODEL_PATH = "model/bert_finetuned/bert_finetuned.onnx";
    private static final int MAX_LENGTH = 128;

    private static OrtEnvironment env;
    private static OrtSession session;
    private static HuggingFaceTokenizer tokenizer;

    static {
        try {
            env = OrtEnvironment.getEnvironment();
            session = createSession();

            tokenizer = HuggingFaceTokenizer.newInstance(Paths.get("src/main/resources/tokenizer/tokenizer.json"));


            System.out.println("Modelo e tokenizer carregados.");
        } catch (Exception e) {
            throw new RuntimeException("Erro ao inicializar BertValidator", e);
        }
    }

    private static OrtSession createSession() throws Exception {
        String modelPath = BertValidator.class.getClassLoader().getResource(MODEL_PATH).getPath();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        return env.createSession(modelPath, options);
    }

    public static int predict(String text) throws Exception {
        text = text.toLowerCase().trim();

        // Tokenizar texto - retorna int[] com os IDs dos tokens
        Encoding encoding = tokenizer.encode(text);
        long[] inputIdsRaw = encoding.getIds();

        // Truncar/pad
        long[] inputIds = new long[MAX_LENGTH];
        long[] attentionMask = new long[MAX_LENGTH];

        int len = Math.min(inputIdsRaw.length, MAX_LENGTH);
        System.arraycopy(inputIdsRaw, 0, inputIds, 0, len);

        for (int i = 0; i < len; i++) {
            attentionMask[i] = 1;
        }

        try (OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), new long[]{1, MAX_LENGTH});
             OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), new long[]{1, MAX_LENGTH})) {

            HashMap<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", inputIdsTensor);
            inputs.put("attention_mask", attentionMaskTensor);

            try (OrtSession.Result result = session.run(inputs)) {
                float[][] logits = (float[][]) result.get(0).getValue();
                System.out.println("Logits: " + Arrays.toString(logits[0]));
                return argMax(logits[0]);
            }
        }
    }

    private static int argMax(float[] array) {
        int maxIdx = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    public static void close() {
        try {
            session.close();
            env.close();
        } catch (Exception e) {
            System.err.println("Erro ao liberar recursos ONNX: " + e.getMessage());
        }
    }

    public static void main(String[] args) throws Exception {
        String text = "Termo de Cessão de Direitos de Cota de Consórcio. A cedente, Beatriz Cristina Ferreira, transfere ao cessionário, Fundo GHI, a cota 4671 do grupo 19783, com tarifa de cessão de R$1762,00, conforme contrato de adesão da Rodobens Consórcios. O cessionário assume 77% das parcelas, em acordo firmado em 12/08/2025 na cidade de Rio de Janeiro, RJ, sob as condições do contrato original.";
        int result = predict(text);
        System.out.println("Classe predita: " + result);
        close();
    }
}
