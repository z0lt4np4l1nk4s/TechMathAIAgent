import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
from common.config import DEVICE, GENERATOR_MODEL_NAME
from utils.logging import log_info
from common.tags import Tags
from common.intents import Intents
from utils.code_executor import execute_python_code, silent_execute
from core.search import SearchEngine

class StopOnTokens(StoppingCriteria):
    """
    Custom stopping criteria to prevent the model from generating 
    extra turns or hallucinating user inputs.
    """
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the last generated token is in our list of stop IDs
        for stop_id in self.stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class LocalGenerator:
    def __init__(self, model_id=GENERATOR_MODEL_NAME, vector_store=None, embedder=None):
        """
        Initializes the local LLM using a non-gated variant.
        Configures 4-bit quantization to fit the model within 6GB VRAM.
        """
        # BitsAndBytes configuration for high-efficiency 4-bit loading
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16, # Use float16 for computation
            bnb_4bit_quant_type="nf4",            # Normal Float 4 quantization
            bnb_4bit_use_double_quant=True,       # Second quantization for extra memory savings
            bnb_4bit_quant_storage=torch.float16
        )

        self.vector_store = vector_store
        self.embedder = embedder
        # Initialize the Search Engine for the RAG pipeline
        self.search_engine = SearchEngine(vector_store=vector_store, embedder=embedder)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Words that indicate the model should stop talking
        self.stop_words = [f"{Tags.USER}", "User:", f"{Tags.SYSTEM}", "PRIMJER", "###"]
        
        # Load the Causal Language Model with memory optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=DEVICE,              # Auto-assign layers to GPU/CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",     # Scaled Dot Product Attention (faster)
            torch_dtype=torch.float16
        )
        self.model.config.use_cache = True

        # Map stop words to specific token IDs for the generator
        self.stop_token_ids = [
            self.tokenizer.eos_token_id,
            self.tokenizer.encode("User:", add_special_tokens=False)[0],
            self.tokenizer.encode(f"{Tags.USER}:", add_special_tokens=False)[0],
            self.tokenizer.encode(f"{Tags.SYSTEM}:", add_special_tokens=False)[0],
            self.tokenizer.convert_tokens_to_ids(Tags.SENTENCE_END)
        ]

    def generate_answer(self, query):
        """
        The main RAG entry point. 
        Detects intent, retrieves documents if necessary, generates a response,
        and executes Python code for math verification.
        """
        log_info("Detecting intent of the query...")
        intent = self._get_intent(query)
        log_info(f"Intent detected: {intent}")

        context = ""
        sources = []

        # RETRIEVAL: If intent is DOCS, query the vector database for context
        if Intents.DOCS in intent and self.search_engine:
            log_info("Retrieving relevant documents...")
            sources = self.search_engine.query(query)
            context = "\n".join(chunk["text"] for chunk in sources)

        # Build the dynamic prompt based on intent and retrieved context
        prompt = self._get_prompt(intent, query, context)

        # Tokenize inputs and move them to the configured DEVICE (GPU)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

        isDocs = Intents.DOCS in intent
        # Grant more tokens for document-based answers than general chat/math
        maxTokens = 768 if isDocs else 512

        # Start the LLM generation process
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=maxTokens,
            do_sample=isDocs,          # Use sampling for variety in DOCS intent
            temperature=0.1 if not isDocs else None, # Low temp for precision in MATH/CODE
            top_p=0.90,
            repetition_penalty=1.2,    # Discourage word loops
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        # Convert token IDs back to human-readable text
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        log_info("FULL decoded output:\n" + decoded)
        
        # Clean the output to only include the assistant's new response
        assistant_text = self._extract_assistant_text(decoded)

        # Wrap code intent responses in Markdown blocks if they aren't already
        if Intents.CODE in intent:
            assistant_text = "```python\n" + self._extract_python_code(assistant_text) + "\n```"

        log_info("Assistant text:\n" + assistant_text)

        # Track if the model actually used the provided RAG context
        was_source_used = Tags.CONTEXT_USED in assistant_text
        assistant_text = assistant_text.replace(Tags.CONTEXT_USED, "").strip()

        # --------------------
        # MATH Logic: Execute the generated Python code
        # --------------------
        if Intents.MATH in intent:
            # Extract the code block using Regex
            code_match = re.search(r"```python\s*(.*?)\s*```", assistant_text, re.DOTALL)

            if not code_match:
                return "Greška: Nije pronađen Python kod u odgovoru modela.", sources, was_source_used

            python_code = code_match.group(1).strip()
            log_info("Executing Python code:\n" + python_code)

            # Perform the mathematical calculation using the safe executor
            execution_result = silent_execute(python_code)

            # Format the final response showing the code and the verified result
            final_answer = (
                "```python\n"
                f"{python_code}\n"
                "```\n\n"
                "**Rezultat izračuna:**\n"
                f"{execution_result}"
            )

            return final_answer, sources, was_source_used

        # --------------------
        # CODE / DOCS / CHAT Logic: Clean and return text
        # --------------------
        clean_answer = assistant_text.strip()

        # Safety split to ensure the model doesn't continue generating into a new task
        clean_answer = re.split(r"\n\s*\[INST\]|\n\s*ZADATAK:", clean_answer)[0].strip()

        log_info("Final answer prepared:\n" + clean_answer)

        return clean_answer, sources, was_source_used
    
    def _get_intent(self, query):
        """
        Detects the intent of the query: MATH, CODE, DOCS, or CHAT.
        Uses a hybrid approach: heuristic (keyword matching) and LLM classification.
        """
        # Step 1: Heuristic check for Document Retrieval (DOCS) intent
        docs_indicators = [
                "zadatak", "zadaci", "zadatke", "primjer", "vježba", "tekst", 
                "iz dokumenta", "pronađi zadatke", "daj mi zadatke"
        ]
        query_l = query.lower()
        if any(indicator in query_l for indicator in docs_indicators):
            return Intents.DOCS
        
        # Step 2: Heuristic check for general programming (CODE) intent
        code_indicators = [
            "napiši funkciju", "napiši kod", "kako napisati", "kako implementirati",
            "napravi funkciju", "napravi kod", "dajte mi kod", "dajte mi funkciju"
        ]
        if any(indicator in query_l for indicator in code_indicators):
            return Intents.CODE

        # Step 3: LLM-based classification for ambiguous queries
        intent_prompt = (
            f"{Tags.SENTENCE_START}{Tags.INST_START}\n"
            f"Analiziraj upit: '{query}'\n"
            "KLJUČNO PRAVILO:\n"
            f"- Ako korisnik pita 'ŠTO JE', 'DEFINIRAJ', 'OBJASNI' ili traži informaciju iz teksta -> {Intents.DOCS}\n"
            f"- Ako korisnik traži 'IZRAČUNAJ', 'RIJEŠI' ili daje matematički zadatak s brojevima -> {Intents.MATH}\n\n"
            "Primjeri upita i njihovih namjera:\n"
            f"- 'Što je metoda konjugiranih gradijenata?' -> {Intents.DOCS}\n"
            f"- 'Definiraj pojam Eulerova formula' -> {Intents.DOCS}\n"    
            f"- 'Izračunaj nultočke ...' -> {Intents.MATH}\n"
            f"- 'Riješi jednadžbu x^2 - 4 = 0' -> {Intents.MATH}\n"
            f"- 'Napiši Python funkciju za sumu liste' -> {Intents.CODE}\n"
            f"- 'Bok, kako si?' -> {Intents.CHAT}\n"
            "Odgovori samo jednom riječi koja najbolje opisuje namjeru:\n"
            f"- {Intents.MATH}: ako se traži numerički izračun ili rješenjavanje zadatka.\n"
            f"- {Intents.CODE}: ako se traži pisanje koda, funkcije ili algoritma.\n"
            f"- {Intents.DOCS}: ako je upit stručni pojam, definicija ili pitanje o dokumentaciji.\n"
            f"- {Intents.CHAT}: pozdrav ili opći razgovor.\n"
            f"Vrati samo jednu riječ od ponuđenih koja najbolje opisuje namjeru, ako nisi siguran, vrati {Intents.CHAT}.\n"
            f"{Tags.INST_END}\n"
            f"{Tags.ANSWER}:\n"
        )
        
        # Tokenize and move to GPU
        inputs = self.tokenizer(intent_prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        
        # Generate a very short output (just the intent keyword)
        output = self.model.generate(**inputs, max_new_tokens=3, do_sample=False, use_cache=False)

        # Decode only the newly generated tokens
        intent = self.tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip().upper()
        
        # Fallback to CHAT if the model output is unexpected
        if intent not in [Intents.MATH, Intents.CODE, Intents.DOCS, Intents.CHAT]:
            intent = Intents.CHAT

        return intent
    
    def _get_prompt(self, intent, query, context = ""):
        """
        Builds a specific system prompt based on the detected intent.
        Each branch defines the behavior and formatting rules for the LLM.
        """
        # MATH INTENT: Focus on generating executable Python code using libraries like NumPy/Pandas
        if Intents.MATH in intent:
            prompt = (
                f"{Tags.SENTENCE_START}{Tags.INST_START}\n"
                "Ti si matematički Python programer.\n"
                "PRAVILA:\n"
                "- Piši ISKLJUČIVO čitljiv i optimiziran Python kod\n"
                "- Pripazi da se kod može izvršiti bez grešaka\n"
                "- Kod mora riješiti zadatak točno\n"
                "- Na raspolaganju su ti biblioteke ako ih trebaš: numpy, pandas\n"
                "- Ne piši objašnjenja\n"
                "- Konačni rezultat spremi u varijablu `result`\n"
                "- Ako postoji više rješenja, spremi SVA u `result` kao listu, nemoj birati samo jedno\n"
                "- `result` može biti broj, string, lista, numpy array ili pandas DataFrame\n"
                "FORMAT: result = <vrijednost>\n\n"
                f"{Tags.TASK}:\n{query}\n"
                f"{Tags.INST_END}\n"
                f"{Tags.ANSWER}:\n"
                "```python\n"
            )
        
        # CODE INTENT: Focus on general programming tasks with Croatian comments
        elif Intents.CODE in intent:
            prompt = (
                f"{Tags.SENTENCE_START}{Tags.INST_START}\n"
                "Ti si stručni programer za Python koji zna pisati samo kod i komentirati ga na hrvatskom.\n"
                "PRAVILA:\n"
                "- Piši ISKLJUČIVO čitljiv i optimiziran kod\n"
                "- Dodaj komentare u kod na hrvatskom\n"
                "- NE dodaj objašnjenje, opis ili primjer\n"
                f"{Tags.INST_END}\n"
                f"# {Tags.TASK}:\n{query}\n"
                f"{Tags.ANSWER}:\n"
                "```python\n"
            )
        
        # DOCS INTENT: Strict Retrieval-Augmented Generation (RAG) with LaTeX formatting
        elif Intents.DOCS in intent:
            prompt = (
                f"{Tags.SENTENCE_START}{Tags.INST_START}\n"
                "ZADATAK: Odgovori isključivo na osnovu KONTEKSTA.\n"
                f"KONTEKST:\n{context}\n\n"
                f"UPIT: {query}\n\n"
                "STROGA PRAVILA:\n"
                f"1. Ako je info u kontekstu, započni odgovor točno sa: {Tags.CONTEXT_USED}\n"
                "2. Ako info NEMA, odgovori: 'Nažalost, dokumenti ne sadrže informacije o tom pojmu.'\n"
                "3. Ne koristi vanjsko znanje. Ako su u kontekstu zadaci, PREPIŠI IH točno kako su navedeni.\n"
                "4. Sve matematičke izraze i matrice OBAVEZNO piši u LaTeX formatu koristeći $inline$ ili $$display$$ (npr. $A = \\begin{pmatrix} ... \\end{pmatrix}$). Pripazi da sve bude čitljivo\n"
                "5. Između teksta i formule OBAVEZNO ostavi jedan prazan red.\n"
                "6. Odgovaraj isključivo na hrvatskom jeziku."
                f"{Tags.INST_END}\n"
                f"{Tags.ANSWER}: {Tags.CONTEXT_USED}"
            )
        
        # CHAT INTENT: General conversational behavior
        else:
            prompt = (
                f"{Tags.SENTENCE_START}{Tags.INST_START}\n"
                "Ti si konverzacijski asistent koji odgovara na hrvatskom jeziku.\n"
                "PRAVILA:\n"
                "- Odgovori ISKLJUČIVO jednom rečenicom\n"
                "- Ne ponavljaj upute ni tekst zadatka\n"
                "- Ne objašnjavaj pojmove\n"
                "- Ne tumači stručne ili tehničke pojmove\n"
                "- Ako upit nije pitanje, zatraži pojašnjenje\n\n"
                f"{Tags.TASK}:\n{query}\n\n"
                f"{Tags.INST_END}\n"
                f"{Tags.ANSWER}:\n"
            )

        return prompt
    
    def _extract_assistant_text(self, decoded_text: str) -> str:
        """
        Extracts the generated assistant output strictly after the instruction or answer tags.
        Crucial for removing the repeated prompt from the output.
        """
        if Tags.INST_END in decoded_text:
            return decoded_text.split(f"{Tags.INST_END}:", 1)[1].strip()
        if Tags.ANSWER in decoded_text:
            return decoded_text.split(f"{Tags.ANSWER}:", 1)[1].strip()
        return decoded_text.strip()
    
    def _extract_python_code(self, text: str) -> str:
        """
        Uses Regular Expressions to isolate Python code from Markdown code blocks.
        """
        match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()