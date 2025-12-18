"""
Resume upload and management API routes.
"""

import os
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status

from backend.app.config import settings
from backend.app.models.schemas import (
    ResumeUploadResponse,
    ResumeSummary,
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    GeneratedQuestion,
    QuestionType,
)
from backend.app.api.deps import (
    get_retriever,
    get_llm,
    ensure_upload_dir,
    validate_file_extension,
    generate_resume_id,
)
from backend.app.services.llm_service import BaseLLMService
from rag.retriever import InterviewRetriever


router = APIRouter(prefix="/resume", tags=["resume"])


@router.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    retriever: InterviewRetriever = Depends(get_retriever),
):
    """
    Upload and process a resume file.
    
    Supports PDF and TXT files. The resume will be:
    1. Saved to disk
    2. Parsed to extract structured information
    3. Chunked into semantic sections
    4. Embedded in vector database for retrieval
    """
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )
    
    if not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE // (1024*1024)}MB",
        )
    
    # Generate resume ID and save file
    resume_id = generate_resume_id(file.filename)
    upload_dir = ensure_upload_dir()
    file_path = upload_dir / f"{resume_id}{Path(file.filename).suffix}"
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process resume through RAG pipeline
        retriever.process_resume(str(file_path), resume_id)
        
        # Get summary info
        summary = retriever.get_resume_summary(resume_id)
        
        return ResumeUploadResponse(
            resume_id=resume_id,
            filename=file.filename,
            chunks_created=summary["total_chunks"],
            sections=summary["sections"],
            message="Resume processed successfully",
        )
        
    except Exception as e:
        # Clean up file if processing failed
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process resume: {str(e)}",
        )


@router.get("/{resume_id}", response_model=ResumeSummary)
async def get_resume_summary(
    resume_id: str,
    retriever: InterviewRetriever = Depends(get_retriever),
):
    """Get summary information for a processed resume."""
    try:
        summary = retriever.get_resume_summary(resume_id)
        
        if summary["total_chunks"] == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Resume not found: {resume_id}",
            )
        
        # Extract additional info from chunks
        chunks = retriever.embedder.get_all_chunks(resume_id)
        
        skills = []
        experience_count = 0
        education_count = 0
        name = None
        
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            section = metadata.get("section", "")
            
            if section == "skills":
                skills_str = metadata.get("skills_list", "")
                if skills_str:
                    skills = [s.strip() for s in skills_str.split(",")]
            elif section == "experience":
                experience_count += 1
            elif section == "education":
                education_count += 1
            elif section == "overview":
                # Try to extract name from contact info
                contact = metadata.get("contact", {})
                if isinstance(contact, dict):
                    name = contact.get("name")
        
        return ResumeSummary(
            resume_id=resume_id,
            name=name,
            total_chunks=summary["total_chunks"],
            sections=summary["sections"],
            skills=skills[:20],  # Limit to 20 skills
            experience_count=experience_count,
            education_count=education_count,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resume summary: {str(e)}",
        )


@router.delete("/{resume_id}")
async def delete_resume(
    resume_id: str,
    retriever: InterviewRetriever = Depends(get_retriever),
):
    """Delete a resume and its embeddings."""
    try:
        # Delete from vector database
        retriever.embedder._delete_resume(resume_id)
        
        # Delete uploaded file if exists
        upload_dir = ensure_upload_dir()
        for ext in settings.ALLOWED_EXTENSIONS:
            file_path = upload_dir / f"{resume_id}.{ext}"
            if file_path.exists():
                os.remove(file_path)
                break
        
        return {"message": f"Resume {resume_id} deleted successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete resume: {str(e)}",
        )


@router.post("/{resume_id}/generate-questions", response_model=GenerateQuestionsResponse)
async def generate_questions(
    resume_id: str,
    request: GenerateQuestionsRequest,
    retriever: InterviewRetriever = Depends(get_retriever),
    llm: BaseLLMService = Depends(get_llm),
):
    """
    Generate interview questions for a resume.
    
    Uses RAG to retrieve relevant resume context and LLM to generate questions.
    """
    # Verify resume exists
    summary = retriever.get_resume_summary(resume_id)
    if summary["total_chunks"] == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resume not found: {resume_id}",
        )
    
    try:
        # Build prompt using retriever
        prompt = retriever.build_prompt(
            resume_id=resume_id,
            focus_area=request.focus_area,
            question_type=request.question_type.value,
            n_questions=request.num_questions,
        )
        
        # System prompt for question generation
        system_prompt = """You are an expert technical interviewer. Generate specific, 
relevant interview questions based on the candidate's resume. 

Rules:
1. Questions should be directly related to the resume content
2. Include a mix of verification questions and deep-dive questions
3. For technical roles, include system design or coding questions
4. For behavioral questions, focus on specific experiences mentioned
5. Return questions in a numbered list format

Format your response as:
1. [Question 1]
2. [Question 2]
...
"""
        
        # Generate questions using LLM
        response = await llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=1024,
            temperature=0.7,
        )
        
        # Parse questions from response
        questions = parse_questions_from_response(response, request.question_type)
        
        return GenerateQuestionsResponse(
            resume_id=resume_id,
            questions=questions,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate questions: {str(e)}",
        )


def parse_questions_from_response(
    response: str,
    question_type: QuestionType,
) -> List[GeneratedQuestion]:
    """Parse LLM response into structured questions."""
    questions = []
    lines = response.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove numbering (1., 2., -, *, etc.)
        import re
        cleaned = re.sub(r'^[\d]+[.)\-]\s*', '', line)
        cleaned = re.sub(r'^[-*]\s*', '', cleaned)
        cleaned = cleaned.strip()
        
        if cleaned and len(cleaned) > 10:  # Minimum question length
            questions.append(GeneratedQuestion(
                question=cleaned,
                type=question_type,
            ))
    
    return questions


@router.get("/{resume_id}/chunks")
async def get_resume_chunks(
    resume_id: str,
    section: str = None,
    retriever: InterviewRetriever = Depends(get_retriever),
):
    """Get all chunks for a resume, optionally filtered by section."""
    try:
        if section:
            chunks = retriever.embedder.search(
                query="",  # Empty query to get all
                n_results=100,
                resume_id=resume_id,
                section_filter=section,
            )
        else:
            chunks = retriever.embedder.get_all_chunks(resume_id)
        
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No chunks found for resume: {resume_id}",
            )
        
        return {"resume_id": resume_id, "chunks": chunks}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chunks: {str(e)}",
        )