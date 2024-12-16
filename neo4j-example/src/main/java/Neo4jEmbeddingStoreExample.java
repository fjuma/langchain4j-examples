import static dev.langchain4j.model.embedding.onnx.PoolingMode.MEAN;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.onnx.OnnxEmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.neo4j.Neo4jEmbeddingStore;

import java.net.URL;
import java.nio.file.Files;
import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Neo4jEmbeddingStoreExample {

    public static void main(String[] args) {
        EmbeddingStore<TextSegment> embeddingStore = Neo4jEmbeddingStore.builder()
                .withBasicAuth("bolt://localhost:7687", "neo4j", "neo4jpassword")
                .dimension(args.length == 1 ? 384 : 768)
                .textProperty("content")
                .label("entity")
                .indexName(args[0])
                .retrievalQuery("ADD RETRIEVAL QUERY HERE")
                .build();

        EmbeddingModel embeddingModel = null;
        if (args.length == 1) {
            embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        } else {
            try {
                Path tempDir = Paths.get(args[1]);
                URL modelUrl = new URL("https://huggingface.co/Xenova/all-mpnet-base-v2/resolve/main/onnx/model.onnx?download=true");
                Path modelPath = tempDir.resolve("model.onnx");
                if (!Files.exists(tempDir)) {
                    Files.createDirectories(tempDir);
                }
                if (!Files.exists(modelPath)) {
                    Files.copy(modelUrl.openStream(), modelPath, REPLACE_EXISTING);
                }
                URL tokenizerUrl = new URL("https://huggingface.co/Xenova/all-mpnet-base-v2/resolve/main/tokenizer.json?download=true");
                Path tokenizerPath = tempDir.resolve("tokenizer.json");
                if (!Files.exists(tokenizerPath)) {
                    Files.copy(tokenizerUrl.openStream(), tokenizerPath, REPLACE_EXISTING);
                }
                embeddingModel = new OnnxEmbeddingModel(modelPath, tokenizerPath, MEAN);
            } catch (Exception e) {
                // ignored
            }
        }

        String query = "ADD YOUR QUERY HERE";

        TextSegment segment = TextSegment.from(query);
        Embedding embedding = embeddingModel.embed(segment).content();
        EmbeddingSearchRequest request = EmbeddingSearchRequest.builder()
                .queryEmbedding(embedding)
                .maxResults(4)
                .build();
        EmbeddingSearchResult<TextSegment> relevant = embeddingStore.search(request);
        System.out.println("=============================================================================================================");
        System.out.println("RETRIEVED " + relevant.matches().size() + " RESULTS");
        System.out.println("=============================================================================================================");
        relevant
                .matches()
                .forEach(match -> {
                    System.out.println(match.embedded().text());
                    System.out.println(match.score());
                    System.out.println(match.embedded().metadata());
                    System.out.println("=============================================================================================================");

                });
    }
}
