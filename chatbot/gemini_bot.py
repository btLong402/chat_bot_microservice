import os
from typing import List, Optional, Sequence

# Suppress noisy gRPC/ALTS logs (must be set before importing google libs)
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "")

try:
    # Prefer the public Generative AI SDK. Avoid heavy setup at import time.
    import google.generativeai as _genai_pkg  # type: ignore
    _HAS_GENAI = True
except Exception:
    _genai_pkg = None
    _HAS_GENAI = False
try:
    from dotenv import load_dotenv
except Exception:
    # dotenv is optional; provide a no-op fallback so importing this module
    # doesn't fail in minimal environments.
    def load_dotenv():
        return None
from .memory import MemoryManager
from .retriever import RAGRetriever

load_dotenv()

import warnings

if not _HAS_GENAI:
    # don't raise at import time; let the application import the module.
    warnings.warn("google-genai package not installed. Install with: pip install google-genai", RuntimeWarning)

class GeminiBot:
    def __init__(
        self,
        name="La Bàn AI",
        model="gemini-2.5-flash-lite",
        memory_file="chat_history.json",
        retriever=None,
        user_id: Optional[str] = None,
    ):
        self.name = name
        self.user_id = user_id
        self.memory = MemoryManager(memory_file)
        self.retriever = retriever or RAGRetriever()
        if not _HAS_GENAI or _genai_pkg is None:
            raise RuntimeError("google-generativeai package not installed. Install with: pip install google-generativeai")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                _genai_pkg.configure(api_key=api_key)
            except Exception as exc:
                raise RuntimeError(f"google-generativeai configure failed: {exc}")
        else:
            warnings.warn("GEMINI_API_KEY not set. Set it via environment or .env file.", RuntimeWarning)
        try:
            self._genai_model = _genai_pkg.GenerativeModel(model)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize GenerativeModel '{model}': {exc}")
        # maintain legacy attributes for compatibility with older calling code
        self.model = None
        self.chat = None

    @property
    def generative_model(self):
        """Expose the underlying GenerativeModel for external orchestration."""
        return self._genai_model

    def ask(
        self,
        message,
        *,
        use_rag: bool = True,
        history_turns: int = 10,
        top_k: int = 3,
        temperature: Optional[float] = None,
        prompt_template: Optional[str] = None,
        return_details: bool = False,
        context_chunks: Optional[Sequence[str]] = None,
        user_id: Optional[str] = None,
    ):
        """
        Hỏi Gemini với khả năng nối mạch hội thoại thực (start_chat) và tự động RAG.
        """

        active_user_id = user_id or self.user_id

        # ===== [1] RAG Context =====
        if context_chunks is not None:
            context_list = list(context_chunks)
        elif use_rag:
            context_list = self.retriever.retrieve(message, top_k=top_k)
        else:
            context_list = []
        context_block = "\n\n".join(context_list).strip() or "(không có tài liệu tham chiếu phù hợp)"

        # ===== [2] Lịch sử hội thoại =====
        conversation = self.memory.get_conversation(user_id=active_user_id) or []
        history_for_prompt = []
        for role, content in conversation[-history_turns:]:
            label = "Người dùng" if role == "user" else self.name
            history_for_prompt.append(f"{label}: {content}")
        history_block = "\n".join(history_for_prompt).strip() or "(không có lịch sử phù hợp)"

        # ===== [3] Nếu có prompt template tùy chỉnh =====
        if prompt_template:
            try:
                prompt = prompt_template.format(
                    assistant_name=self.name,
                    query=message,
                    history_block=history_block,
                    context_block=context_block,
                    history=history_block,
                    context=context_block,
                )
            except Exception as exc:
                raise ValueError(f"Invalid prompt_template: {exc}")
        else:
            # Prompt chuẩn (mặc định)
           prompt = f""" [1] Vai trò (Role): Bạn là **Du học Mentor Pro** (tên: {self.name}) — Trợ lý AI tư vấn du học chuyên nghiệp cho học sinh – sinh viên Việt Nam. Hiểu rõ về ngành, học bổng, visa, chi phí và điều kiện đầu vào tại các nước phổ biến (Mỹ, Canada, Úc, Anh, Nhật, Hàn, Singapore...). **Phong cách:** thực tế, logic, không suy diễn cảm xúc, chỉ phản hồi dựa trên dữ liệu có trong input. [2] Mục tiêu (Goal): Tạo phản hồi ngắn gọn, định hướng và hành động rõ — giúp người dùng xác định hướng đi du học phù hợp nhất. [3A] Ngữ cảnh dữ liệu (Context Data): - Tin nhắn: {message} - Tài liệu tham chiếu (context_block): {context_block} **Cách dùng context_block:** - Dùng *chỉ khi* nội dung người dùng chứa ≥2 từ khóa trùng với context (ví dụ: tên trường, ngành, năm). - Nếu message không liên quan (chào hỏi / ngoài chủ đề du học) → bỏ qua context_block. - Nếu có mâu thuẫn giữa message và context_block → ưu tiên message hiện tại và ghi chú: “Lưu ý: thông tin trong context khác với yêu cầu mới — dùng dữ liệu mới để phân tích.” [3B] Ngữ cảnh hội thoại (Conversation History): {history_block} **Cách dùng history_block:** - Nếu trống hoặc không liên quan → xem là cuộc trò chuyện mới. - Nếu đã có hội thoại → không chào lại, trả lời tiếp mạch lạc. - Nếu chứa quyết định trước (quốc gia/ngành đã chọn) → tóm tắt ngắn trước khi đề xuất, không hỏi lại. - Nếu mâu thuẫn → ưu tiên yêu cầu mới và báo rõ. [4] Quy tắc & Phong cách (Rules & Style): - Ngôn ngữ: tiếng Việt thân thiện, rõ ràng, đúng ngữ pháp. - Phản hồi theo mạch tự nhiên, không ép cấu trúc, nhưng vẫn có trình tự hợp lý: (1) Nắm bắt và xác nhận yêu cầu. (2) Phân tích 2–3 điểm trọng tâm (điều kiện, cơ hội, lưu ý). (3) Đưa hướng hành động hoặc câu hỏi tiếp theo. - Tránh liệt kê khô cứng; ưu tiên diễn đạt như mentor hướng dẫn thực tế. 📍**Nếu thiếu dữ liệu:** - Hỏi tối đa 3 câu ngắn, theo **thứ tự ưu tiên**: (1) Bậc học → (2) Ngành → (3) Quốc gia → (4) Ngân sách → (5) Hạn/visa. - Khi user đã cho 1 mục, bỏ qua câu đó, hỏi tiếp mục kế tiếp. - Mẫu: "Để gợi ý nhanh: 1) Bậc học? 2) Ngành? 3) Ngân sách (VND/năm)?" 📍**Nếu user hỏi ngoài phạm vi du học:** - Trả fallback: "Hiện tôi chuyên tư vấn du học. Với câu hỏi này, tôi có thể: (A) chuyển sang chủ đề liên quan du học nếu có, hoặc (B) gợi ý nguồn/từ khóa để bạn tra cứu — bạn chọn A hay B?" - Nếu chủ đề pháp lý / y tế / tài chính chuyên sâu → cảnh báo: "Tôi không phải luật sư/bác sĩ; bạn nên tham khảo chuyên gia chính thức." 📍**Tone chào đầu (nếu là tin nhắn đầu tiên):** Chọn ngẫu nhiên 1 trong 4 biến thể: 1. "Chào (name)! Bạn đang cân nhắc du học bậc nào?" 2. "Xin chào — nói tôi nghe 2 điều bạn quan tâm nhất về du học nhé (quốc gia & ngành)?" 3. "Chào bạn, tôi có thể giúp chọn trường hoặc check học bổng — bạn muốn gì trước?" 4. "Rất vui được giúp! Bạn đã có quốc gia hoặc ngành học trong đầu chưa?" 📍**Edge cases:** - User chỉ chào / small talk → đáp 1 câu + CTA: "Bạn quan tâm điều gì về du học?" - User gửi file CV / transcript → tóm tắt 3 điểm mạnh rồi hỏi 2 câu ưu tiên. - User hỏi tính chi phí → đưa khoảng giá + giả định + ghi chú nguồn. - User xin danh sách trường → top 5, có lý do + tiêu chí rõ ràng. 📍**Độ tin cậy (Confidence Tag):** - Cao (≥80%): nguồn chính phủ / đại học / tổ chức học bổng chính thức. - Trung bình (50–80%): nguồn blog hoặc diễn đàn có kiểm chứng. - Thấp (<50%): suy luận / ước lượng, phải ghi rõ giả định. → Khi nêu số liệu, thêm ví dụ: “Tin cậy: 85% — theo website chính phủ Canada, cập nhật 03/2025.” [5] Định dạng đầu ra (Output Format): - Nếu **tin nhắn đầu tiên** (history_block trống) → chào ngắn gọn (1 trong 4 mẫu trên). - Nếu **đã có hội thoại** → không chào lại, trả lời trực tiếp. - Nếu **có đủ dữ kiện** → phản hồi theo 4 phần chuẩn. - Nếu **thiếu dữ kiện** → kích hoạt chế độ hỏi ngắn (≤4 câu). - Nếu **ngoài phạm vi du học** → dùng fallback định dạng chuẩn. [6] Kiểm chứng & Đối chiếu (Validation): - So sánh dữ liệu giữa {message}, {context_block}, {history_block}. - Ưu tiên dữ liệu liên kết, bỏ dữ liệu mâu thuẫn. - Nếu chưa đủ dữ liệu → nói rõ “chưa đủ dữ kiện để kết luận”. - Với số liệu cụ thể (chi phí, điểm, deadline) → ghi rõ phạm vi + nguồn. - Báo “Tin cậy: XX% — theo nguồn Y (tháng/năm)” hoặc “Ước lượng nếu không có nguồn”. """

        # ===== [4] Sinh phản hồi với cơ chế nối mạch =====
        generation_kwargs = {"generation_config": {"temperature": temperature}} if temperature else {}

        try:
            if conversation:
                # Có lịch sử hội thoại → dùng start_chat()
                if not self.chat:
                    # Tạo chat session từ lịch sử
                    formatted_history = [
                        {"role": "user" if r == "user" else "model", "parts": c}
                        for r, c in conversation[-history_turns:]
                    ]
                    self.chat = self._genai_model.start_chat(history=formatted_history)
                # Gửi tin nhắn mới nối mạch
                response = self.chat.send_message(message, **generation_kwargs)
                answer = response.text.strip()
            else:
                # Không có lịch sử → lần đầu (dùng generate_content)
                response = self._genai_model.generate_content(prompt, **generation_kwargs)
                answer = getattr(response, "text", str(response)).strip()

        except Exception as e:
            raise RuntimeError(f"Gemini request failed: {e}")

        # ===== [5] Cập nhật bộ nhớ =====
        self.memory.add_message("user", message, user_id=active_user_id)
        self.memory.add_message("assistant", answer, user_id=active_user_id)

        # ===== [6] Trả kết quả =====
        if return_details:
            return {
                "answer": answer,
                "context": context_list,
                "prompt": prompt,
                "history_block": history_block,
            }
        return answer

    def clear_context(self):
        self.memory.clear_history(user_id=self.user_id)
        # No persistent chat object when using the `google-genai` client
        # in this simplified adapter; just clear conversation memory.
        self.chat = None
        return "🧹 Bộ nhớ hội thoại đã được xóa!"

