package com.example.validacaodocumento;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.*;
import java.nio.LongBuffer;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Classe para validar documentos usando um modelo BERT treinado exportado em formato ONNX.
 */
public class BertValidatorGrok {
    private static final String MODEL_PATH = "model/bert_finetuned/bert_finetuned.onnx";
    private static final String TOKENIZER_PATH = "tokenizer/tokenizer.json";
    private static final int MAX_LENGTH = 256;

    private static OrtEnvironment env;
    private static OrtSession session;
    private static HuggingFaceTokenizer tokenizer;

    static {
        try {
            if (BertValidatorGrok.class.getClassLoader().getResource(MODEL_PATH) == null) {
                throw new IllegalStateException("Arquivo ONNX não encontrado no classpath: " + MODEL_PATH);
            }
            if (BertValidatorGrok.class.getClassLoader().getResource(TOKENIZER_PATH) == null) {
                throw new IllegalStateException("Arquivo tokenizer.json não encontrado no classpath: " + TOKENIZER_PATH);
            }

            env = OrtEnvironment.getEnvironment();
            session = createSession();
            tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(BertValidatorGrok.class.getClassLoader().getResource(TOKENIZER_PATH).toURI()));
            System.out.println("Modelo e tokenizer carregados.");

            // Registrar shutdown hook
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                try {
                    close();
                } catch (Exception e) {
                    System.err.println("Erro ao fechar recursos: " + e.getMessage());
                }
            }));
        } catch (Exception e) {
            throw new RuntimeException("Erro ao inicializar BertValidatorGrok", e);
        }
    }

    private static OrtSession createSession() throws Exception {
        String modelPath = BertValidatorGrok.class.getClassLoader().getResource(MODEL_PATH).getPath();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        return env.createSession(modelPath, options);
    }

    /**
     * Representa o resultado da predição, incluindo a classe predita e as probabilidades.
     */
    public static class Prediction {
        private final int predictedClass;
        private final float[] probabilities;

        public Prediction(int predictedClass, float[] probabilities) {
            this.predictedClass = predictedClass;
            this.probabilities = probabilities;
        }

        public int getPredictedClass() { return predictedClass; }
        public float[] getProbabilities() { return probabilities; }
    }

    /**
     * Realiza a predição da classe de um texto usando o modelo BERT.
     * @param text O texto a ser classificado (não pode ser nulo ou vazio).
     * @return Um objeto Prediction com a classe predita e as probabilidades.
     * @throws Exception Se houver erro durante a inferência.
     */
    public static Prediction predict(String text) throws Exception {
        if (text == null || text.trim().isEmpty()) {
            throw new IllegalArgumentException("O texto de entrada não pode ser nulo ou vazio.");
        }

        text = text.toLowerCase().trim();
        Encoding encoding = tokenizer.encode(text);
        long[] inputIdsRaw = encoding.getIds();
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
                @SuppressWarnings("unchecked")
                float[][] logits = (float[][]) result.get(0).getValue();
                System.out.println("Logits: " + Arrays.toString(logits[0]));
                float[] probabilities = softmax(logits[0]);
                return new Prediction(argMax(logits[0]), probabilities);
            }
        }
    }

    /**
     * Realiza a predição para documentos longos, dividindo em trechos.
     * @param text O texto do documento.
     * @return A classe predita (baseada na maior probabilidade média dos trechos).
     * @throws Exception Se houver erro durante a inferência.
     */
    public static Prediction predictLongDocument(String text) throws Exception {
        List<String> chunks = splitIntoChunks(text, MAX_LENGTH - 10);
        List<Integer> chunkPredictions = new ArrayList<>();
        List<float[]> chunkProbabilities = new ArrayList<>();
        List<Float> weights = new ArrayList<>();
        for (String chunk : chunks) {
            Prediction prediction = predict(chunk);
            System.out.println("Trecho: " + chunk);
            System.out.println("Classe predita para o trecho: " + prediction.getPredictedClass());
            System.out.println("Probabilidades para o trecho: " + Arrays.toString(prediction.getProbabilities()));
            chunkPredictions.add(prediction.getPredictedClass());
            chunkProbabilities.add(prediction.getProbabilities());
            // Peso baseado na maior probabilidade
            float maxProb = Math.max(prediction.getProbabilities()[0], prediction.getProbabilities()[1]);
            weights.add(maxProb);
        }
        // Calcular a média das probabilidades ponderada pelos pesos
        float[] weightedProbs = new float[2];
        float totalWeight = 0;
        for (int i = 0; i < chunks.size(); i++) {
            float weight = weights.get(i);
            totalWeight += weight;
            for (int j = 0; j < 2; j++) {
                weightedProbs[j] += chunkProbabilities.get(i)[j] * weight;
            }
        }
        float[] avgProbabilities = new float[2];
        for (int i = 0; i < 2; i++) {
            avgProbabilities[i] = weightedProbs[i] / totalWeight;
        }
        int predictedClass = avgProbabilities[0] > avgProbabilities[1] ? 0 : 1;
        return new Prediction(predictedClass, avgProbabilities);
    }

    private static List<String> splitIntoChunks(String text, int chunkSize) {
        List<String> chunks = new ArrayList<>();
        String[] words = text.split("\\s+");
        StringBuilder chunk = new StringBuilder();
        int count = 0;
        for (String word : words) {
            if (count + word.length() + 1 > chunkSize) {
                chunks.add(chunk.toString().trim());
                chunk = new StringBuilder();
                count = 0;
            }
            chunk.append(word).append(" ");
            count += word.length() + 1;
        }
        if (chunk.length() > 0) {
            chunks.add(chunk.toString().trim());
        }
        return chunks;
    }

    private static float[] softmax(float[] logits) {
        float max = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > max) {
                max = logits[i];
            }
        }
        float sum = 0;
        float[] probabilities = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = (float) Math.exp(logits[i] - max);
            sum += probabilities[i];
        }
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }
        return probabilities;
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

    /**
     * Fecha os recursos do ONNX Runtime.
     */
    public static void close() {
        try {
            session.close();
            env.close();
        } catch (Exception e) {
            System.err.println("Erro ao liberar recursos ONNX: " + e.getMessage());
        }
    }

    public static void main(String[] args) throws Exception {
        //String text = "--- Página 1 --- DocuSign Envelope ID: 082146A0-07F7-4AF0-99B6-591DD8A89B1F Aditamento ao Contrato de Participação em Grupo de Consórcio de Bem Móvel/Imóvel -— Cessão de Direitos e Obrigações 1. Administradora/Credora Fiduciária Itaú Adm de Consórcio Ltda com sede na Alameda Pedro Calil, 43, Centro, Poá, SP, inscrita no CNPJ sob o nº 00.000.776-0001-01. 2. Cedente . 2.1 Nome 2.2 CPFÍICNPJ MARIA DA CONSOLAÇÃO FREITAS 009.678.466-09 3. Cessionário 3.1 Nome 3.2 CPFÍCNPJ Fundo de Investimentos em Direitos Creditórios Consorciei II 43.237.790/0001-36 3.3 EndereçoiSede 3.4 Dados da Conta Corrente Rua Gomes de Carvalho, nº. 1195, 4º andar, São Paulo/SP Agência Conta - DAC 3.5. Telefones do cessionário: (11) 3500-7238 4. Dados do Contrato 4.1 Grupo 4.2 Cota 4.3 Percentual Pago 4.4 Percentual a Vencer 20280 283 38.71% 61.29% 5.Modo de Pagamento 6.Tarifa de Aditamento Contratual 5.1 (X) documento de cobrança R$ 650,00 5.2( ) débito na conta corrente indicada no subitem A empresa indicada no item 1, doravante designada Administradora, e as pessoas qualificadas nos itens 2 e 3, designadas respectivamente, Cedente e Cessionário, aditam o Contrato de Participação em Grupo de Consórcio de Bem Móvelilmóvel (“Contrato de Adesão”) indicado no item 4, de acordo com as cláusulas que seguem. 7. Objeto - O Cedente, autorizado pela Administradora, transfere ao Cessionário todos direitos e obrigações previstos no Contrato de Adesão que regula o Grupo/Cota de consórcio apontados no item 4. 7.1 0 CESSIONÁRIO DECLARA QUE RECEBEU CÓPIA DO CONTRATO DE ADESÃO, O LEU PREVIAMENTE. NÃO TENDO QUALQUER DÚVIDA EM RELAÇÃO AO QUE ALI RESTOU DISPOSTO. CONCORDA COM TODAS AS CLÁUSULAS E CONDIÇÕES DO CONTRATO DE ADESÃO E. ATRAVÉS DESTE INSTRUMENTO, ASSUME TODAS AS OBRIGAÇÕES NELE PREVISTAS, EM ESPECIAL AS DE NATUREZA FINANCEIRA. 8. Modo de Pagamento - O Cessionário fará os pagamentos na forma indicada no item 5. 8.1. Sendo assinalado o subitem 5.2, o Cessionário pagará todos os valores devidos em decorrência do Contrato de Adesão mediante débito em sua conta corrente mantida junto ao Itaú Unibanco S.A., indicada no subitem 3.4, que deverá ter saldo disponível suficiente. O Itaú Unibanco S.A., desde já autorizado a efetuar o débito, entregará o respectivo valor à Administradora. 8.2 A insuficiência de saldo na conta corrente apontada no subitem 3.4 configurará atraso no pagamento. 8.3 Sendo indicado o subitem 5.1, o Cessionário fará todos os pagamentos por meio de documento de cobrança (camê ou equivalente) a ser emitido pelo Itaú Unibanco S.A. e enviado para o endereço indicado no subitem 3.3. Se o Cessionário não receber o documento de cobrança em tempo hábil para efetuar o pagamento, deverá comunicar o fato à Administradora, que o instruirá a respeito de como proceder. Em nenhuma hipótese o não recebimento do documento de cobrança eximirá o Cessionário do pagamento. 8.4 O Cessionário está ciente de que. havendo atraso no pagamento de qualquer parcela. ficará impedido de concorrer aos sorteios e de ofertar lances, sem prejuizo das demais sanções previstas no Contrato de Adesão. 9. Tolerância — A tolerância das partes quanto ao descumprimento de qualquer obrigação pela outra parte não significará renúncia ao direito de exigir o cumprimento da obrigação, nem perdão, nem alteração do que foi aqui ajustado. 10.Tarifa - O Cessionário pagará a taxa indicada no item 6 e prevista no Contrato de Adesão, em razão desta SEX - Baik Otiie IICVO --- Página 2 --- DocuSign Envelope ID: 082146A0-07F7-4AF0-99B6-591DD8A89B1F cessão de direitos e obrigações. 11. O CESSIONÁRIO, NESTE ATO, CONFERE PODERES ESPECIAIS À ADMINISTRADORA, NA QUALIDADE DE GESTORA DOS NEGÓCIOS DO GRUPO E MANDATÁRIA DOS SEUS INTERESSES E DIREITOS, PARA, EM CARÁTER IRREVOGÁVEL E IRRETRATÁVEL, (I) TOMAR TODAS AS PROVIDÊNCIAS NECESSÁRIAS À ADMINISTRAÇÃO DO GRUPO. INCLUSIVE RECEBER E DAR QUITAÇÃO, EFETUAR PAGAMENTOS E CONSTITUIR ADVOGADOS PARA A DEFESA DOS INTERESSES DO GRUPO; (Il) REPRESENTÁ-LO PERANTE OUTROS CONSORCIADOS, TERCEIROS, ÓRGÃOS GOVERNAMENTAIS E EMPRESAS SEGURADORAS PARA A CONTRATAÇÃO DOS SEGUROS PREVISTOS NO CONTRATO DE ADESÃO; E (Ill) REPRESENTÁ-LO NAS ASSEMBLEIAS GERAIS ORDINÁRIAS EM QUE NÃO ESTIVER PRESENTE, INCLUSIVE VOTANDO AS MATÉRIAS DA ORDEM DO DIA. 12. O processo de cessão de cota será automaticamente cancelado se estiver em andamento o registro do contrato perante o cartório de registro de imóveis. 13. Cláusulas inalteradas - Permanecem em vigor as disposições do Contrato de Adesão aditado não expressamente alteradas por este Aditamento. 14. Foro - Fica eleito o Foro da Comarca do local da assinatura deste Aditamento, podendo a parte que promover a ação optar pelo Foro do domicílio do Cessionário. 15. Solução Amigável de Conflitos - Para a solução amigável de eventuais conflitos relacionados a este Aditamento, o Cessionário poderá dirigir seu pedido ou reclamação à sua agência Itaú Unibanco S/A. A Administradora coloca ainda à disposição do Cessionário o SAC - Itaú (0800 728 0728). o SAC - Itaú exclusivo ao deficiente auditivo (0800 722 1722) e o Fale Conosco (wwnw.itau.com.br). Se não for solucionado o conflito, o Cessionário poderá recorrer à Ouvidoria Corporativa Itaú (0800 570 0011, em dias úteis das 9h às 18h, Caixa Postal 67.600, CEP 03162-971). E por estarem de acordo, assinam o presente em 03 (três) vias de igual teor e forma, na presença de 02 (duas) testemunhas. Local e Data: São Paulo, 03 de janeiro de 2024 Li (LEMOS) ESTE ADITAMENTO PREVIAMENTE E NÃO TENHO (TEMOS) DÚVIDA SOBRE QUALQUER UMA DE SUAS CLÁUSULAS. DocuSigned by Olsagndae Calumans dog rrta Cessionári (04 Cedente fundo de Investimentos em Direitos Creditórios MARIA DA CONSOLAÇÃO FREITAS onsorcie 3 o) Rataei inocençio À Biencou: RT L5I062.2 cos poa re e Adminiskadora Devedor Solidário a) o) Nome: Nome: CPF CPF Testemunhas a) D) Nome: Nome CPF: CPF: SEX - Baik Otiie IICVO";
        //String text = "--- Página 1 --- CONTRATO DE LOCAÇÃO DE IMÓVEL RESIDENCIAL Pelo presente instrumento particular, as partes: LOCADOR: João da Silva, brasileiro, solteiro, CPF nº 000.000.000-00, residente à Rua das Flores, nº 100, Bairro Centro, Cidade XYZ; LOCATÁRIO: Maria Oliveira, brasileira, solteira, CPF nº 111.111.111-11, residente na Rua dos Lírios, nº 200, Bairro Jardim, Cidade XYZ; Têm entre si, justo e contratado o seguinte: CLÁUSULA PRIMEIRA - DO OBJETO: O LOCADOR dá em locação ao LOCATÁRIO o imóvel residencial situado na Rua das Acácias, nº 300, Bairro Primavera, Cidade XY Z, exclusivamente para fins residenciais. CLÁUSULA SEGUNDA - DO PRAZO: O prazo da locação será de 12 (doze) meses, com início em 01 de maio de 2024 e término em 30 de abril de 2025. CLÁUSULA TERCEIRA - DO VALOR: O aluguel mensal será de RS 1.500,00 (mil e quinhentos reais), com vencimento no dia 5 (cinco) de cada mês. CLÁUSULA QUARTA - DOS ENCARGOS: Serão de responsabilidade do LOCATÁRIO todas as despesas de consumo (água, luz, gás, internet) e taxa de condomínio, se houver. CLÁUSULA QUINTA - DA CONSERVAÇÃO: O LOCATÁRIO compromete-se a manter 0 imóvel em perfeito estado de conservação e a devolvê- lo nas mesmas condições ao final do contrato. E por estarem de pleno acordo, assinam este instrumento em duas vias de igual teor e forma. Cidade XYZ, 01 de maio de 2024. João da Silva — LOCADOR Maria Oliveira — LOCATÁRIA";
        String text = "Acordo de Confidencialidade de Participação em Projeto nº 7223  \\n1. Partes  \\n1.1. Empresa: Pereira Garcia - EI, com sede em Vereda de Garcia, 51, Vila Nova Gameleira 1ª Seção, 41956-877 Freitas do Norte / AC, inscrito no CNPJ sob o nº 67.289.053/0001-45.  \\n1.2. Participante: Rafael Almeida Santos, CPF nº 529.143.806-15, residente em Núcleo Caleb Lima, 81, Jardim América, 79973719 Ribeiro do Sul / AM.  \\n2. Objeto do Acordo  \\nEste acordo tem como objetivo garantir a confidencialidade das informações relacionadas ao Projeto O prazer de evoluir sem preocupação, no qual o Participante atuará como consultor/colaborador.  \\n3. Obrigações de Confidencialidade  \\n3.1. O Participante compromete-se a não divulgar, reproduzir ou utilizar as informações confidenciais do Projeto para qualquer finalidade que não seja a execução do mesmo.  \\n3.2. As informações confidenciais incluem, mas não se limitam a, dados técnicos, financeiros, comerciais e estratégicos do Projeto.  \\n4. Prazo de Confidencialidade  \\nAs obrigações de confidencialidade permanecerão em vigor por 1 anos após o término do Projeto.  \\n5. Penalidades  \\nO descumprimento deste acordo sujeitará o Participante a indenizações por perdas e danos, além de medidas judiciais cabíveis.  \\n6. Foro  \\nFica eleito o foro da comarca de Brasília, DF para dirimir quaisquer dúvidas ou litígios decorrentes deste acordo.  \\nLocal e Data: Brasília, DF, 30/06/2025.  \\nAssinaturas:  \\nEmpresa: ___________________________  \\nParticipante: ___________________________  \\nTestemunhas:  \\n1. Nome: Felipe Augusto Costa CPF: 213.749.865-00  \\n2. Nome: Carla Mendes Ferreira CPF: 942.138.067-31";
        Prediction result = predictLongDocument(text);
        System.out.println("Classe predita: " + result.getPredictedClass());
        System.out.println("Probabilidades: " + Arrays.toString(result.getProbabilities()));
        close();
    }
}