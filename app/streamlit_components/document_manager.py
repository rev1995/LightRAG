"""
Document Manager Component for LightRAG Streamlit Frontend
Upload, manage, and monitor document processing
"""

import streamlit as st
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import json

# Import API client
from utils.api_client import get_api_client


class DocumentManager:
    """Document management component for file uploads and processing"""
    
    def __init__(self):
        self.api_client = get_api_client()
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state for document management"""
        
        if "uploaded_documents" not in st.session_state:
            st.session_state.uploaded_documents = []
        
        if "processing_status" not in st.session_state:
            st.session_state.processing_status = {}
        
        if "upload_progress" not in st.session_state:
            st.session_state.upload_progress = {}
    
    def render(self):
        """Render the complete document management interface"""
        
        st.title("📁 Document Manager")
        
        # Upload section
        self.render_upload_section()
        
        st.markdown("---")
        
        # Document list and management
        self.render_document_list()
        
        st.markdown("---")
        
        # Processing status
        self.render_processing_status()
    
    def render_upload_section(self):
        """Render document upload interface"""
        
        st.subheader("📤 Upload Documents")
        
        # Upload methods
        upload_method = st.radio(
            "Upload Method:",
            ["File Upload", "Text Input", "Bulk Upload"],
            index=0,
            horizontal=True
        )
        
        if upload_method == "File Upload":
            self.render_file_upload()
        elif upload_method == "Text Input":
            self.render_text_input()
        else:
            self.render_bulk_upload()
    
    def render_file_upload(self):
        """Render single file upload interface"""
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=["txt", "pdf", "docx", "md", "csv"],
            accept_multiple_files=True,
            help="Supported formats: TXT, PDF, DOCX, MD, CSV"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"📄 {uploaded_file.name}")
                    st.caption(f"Size: {uploaded_file.size / 1024:.1f} KB")
                
                with col2:
                    if st.button(f"Upload", key=f"upload_{uploaded_file.name}"):
                        self.upload_file(uploaded_file)
                
                with col3:
                    st.write(f"Type: {uploaded_file.type}")
    
    def render_text_input(self):
        """Render text input interface"""
        
        with st.form("text_upload_form"):
            document_name = st.text_input(
                "Document Name",
                placeholder="Enter a name for this document"
            )
            
            document_content = st.text_area(
                "Document Content",
                height=300,
                placeholder="Paste or type your document content here..."
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                submit_button = st.form_submit_button("📝 Upload Text", use_container_width=True)
            
            with col2:
                clear_button = st.form_submit_button("🗑️ Clear", use_container_width=True)
            
            if submit_button and document_content.strip():
                if not document_name.strip():
                    document_name = f"Text_Document_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                self.upload_text_content(document_name, document_content)
            
            if clear_button:
                st.rerun()
    
    def render_bulk_upload(self):
        """Render bulk upload interface"""
        
        st.info("📁 Select a folder containing documents for bulk upload")
        
        # Folder path input
        folder_path = st.text_input(
            "Folder Path",
            placeholder="Enter the path to your documents folder",
            help="Enter the full path to a folder containing documents"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📂 Browse Folder", use_container_width=True):
                if folder_path and os.path.exists(folder_path):
                    self.process_bulk_upload(folder_path)
                else:
                    st.error("Invalid folder path")
        
        with col2:
            if st.button("🔄 Refresh List", use_container_width=True):
                self.refresh_document_list()
    
    def render_document_list(self):
        """Render list of uploaded documents"""
        
        st.subheader("📋 Document Library")
        
        # Refresh button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                self.refresh_document_list()
        
        with col2:
            if st.button("🗑️ Delete All", use_container_width=True):
                if st.session_state.get('confirm_delete_all', False):
                    self.delete_all_documents()
                    st.session_state.confirm_delete_all = False
                else:
                    st.session_state.confirm_delete_all = True
                    st.warning("⚠️ Click again to confirm deletion of all documents")
        
        with col3:
            search_term = st.text_input("🔍 Search documents", placeholder="Search by name...")
        
        # Get document list
        self.refresh_document_list()
        
        if not st.session_state.uploaded_documents:
            st.info("📄 No documents uploaded yet. Upload some documents to get started!")
            return
        
        # Filter documents
        filtered_docs = st.session_state.uploaded_documents
        if search_term:
            filtered_docs = [
                doc for doc in filtered_docs 
                if search_term.lower() in doc.get('name', '').lower()
            ]
        
        # Display documents
        if filtered_docs:
            for doc in filtered_docs:
                self.render_document_item(doc)
        else:
            st.info("No documents match your search.")
    
    def render_document_item(self, doc: Dict[str, Any]):
        """Render individual document item"""
        
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                # Document name and info
                st.write(f"📄 **{doc.get('name', 'Unknown')}**")
                
                # Additional info if available
                if 'size' in doc:
                    st.caption(f"Size: {doc['size']} bytes")
                if 'upload_time' in doc:
                    st.caption(f"Uploaded: {doc['upload_time']}")
            
            with col2:
                # Processing status
                status = doc.get('status', 'unknown')
                if status == 'processed':
                    st.success("✅ Processed")
                elif status == 'processing':
                    st.warning("⏳ Processing")
                elif status == 'failed':
                    st.error("❌ Failed")
                else:
                    st.info("📝 Pending")
            
            with col3:
                # View/Edit button
                if st.button("👁️ View", key=f"view_{doc.get('id', 'unknown')}"):
                    self.view_document(doc)
            
            with col4:
                # Delete button
                if st.button("🗑️ Delete", key=f"delete_{doc.get('id', 'unknown')}"):
                    self.delete_document(doc.get('id'))
            
            st.markdown("---")
    
    def render_processing_status(self):
        """Render document processing status"""
        
        st.subheader("⚙️ Processing Status")
        
        if not st.session_state.processing_status:
            st.info("No documents currently being processed.")
            return
        
        # Show processing progress
        for doc_id, status in st.session_state.processing_status.items():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"Processing: {status.get('name', doc_id)}")
                progress_value = status.get('progress', 0)
                st.progress(progress_value / 100)
            
            with col2:
                st.write(f"{progress_value}%")
                if progress_value >= 100:
                    st.success("Complete")
    
    def upload_file(self, uploaded_file):
        """Upload a single file"""
        
        try:
            # Save file temporarily
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            
            temp_path = temp_dir / uploaded_file.name
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Upload via API
            with st.spinner(f"Uploading {uploaded_file.name}..."):
                result = self.api_client.upload_document(
                    str(temp_path),
                    document_name=uploaded_file.name
                )
            
            # Cleanup temp file
            temp_path.unlink(missing_ok=True)
            
            if result["success"]:
                st.success(f"✅ Successfully uploaded {uploaded_file.name}")
                self.refresh_document_list()
            else:
                st.error(f"❌ Failed to upload {uploaded_file.name}: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"Upload error: {str(e)}")
    
    def upload_text_content(self, name: str, content: str):
        """Upload text content"""
        
        try:
            with st.spinner(f"Uploading {name}..."):
                result = self.api_client.upload_document_content(content, name)
            
            if result["success"]:
                st.success(f"✅ Successfully uploaded {name}")
                self.refresh_document_list()
            else:
                st.error(f"❌ Failed to upload {name}: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"Upload error: {str(e)}")
    
    def process_bulk_upload(self, folder_path: str):
        """Process bulk upload from folder"""
        
        try:
            folder = Path(folder_path)
            
            if not folder.exists():
                st.error("Folder does not exist")
                return
            
            # Find supported files
            supported_extensions = ['.txt', '.pdf', '.docx', '.md', '.csv']
            files = []
            
            for ext in supported_extensions:
                files.extend(folder.glob(f"*{ext}"))
            
            if not files:
                st.warning("No supported files found in the folder")
                return
            
            st.info(f"Found {len(files)} files to upload")
            
            # Upload each file
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file_path in enumerate(files):
                status_text.write(f"Uploading {file_path.name}...")
                
                try:
                    result = self.api_client.upload_document(
                        str(file_path),
                        document_name=file_path.name
                    )
                    
                    if not result["success"]:
                        st.warning(f"Failed to upload {file_path.name}: {result.get('error')}")
                
                except Exception as e:
                    st.warning(f"Error uploading {file_path.name}: {str(e)}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(files))
                time.sleep(0.1)  # Brief pause
            
            status_text.write("✅ Bulk upload completed!")
            self.refresh_document_list()
        
        except Exception as e:
            st.error(f"Bulk upload error: {str(e)}")
    
    def refresh_document_list(self):
        """Refresh the list of uploaded documents"""
        
        try:
            result = self.api_client.list_documents()
            
            if result["success"]:
                st.session_state.uploaded_documents = result["data"]
            else:
                st.error(f"Failed to fetch documents: {result.get('error', 'Unknown error')}")
                st.session_state.uploaded_documents = []
        
        except Exception as e:
            st.error(f"Error fetching documents: {str(e)}")
            st.session_state.uploaded_documents = []
    
    def view_document(self, doc: Dict[str, Any]):
        """View document details"""
        
        with st.expander(f"📄 Document Details: {doc.get('name', 'Unknown')}", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Document Information:**")
                for key, value in doc.items():
                    if key not in ['content', 'embedding']:
                        st.write(f"- **{key.title()}:** {value}")
            
            with col2:
                st.write("**Actions:**")
                
                if st.button("📊 View Processing Status", key=f"status_{doc.get('id')}"):
                    self.check_processing_status(doc.get('id'))
                
                if st.button("🔄 Reprocess", key=f"reprocess_{doc.get('id')}"):
                    self.reprocess_document(doc.get('id'))
                
                if st.button("📥 Export", key=f"export_{doc.get('id')}"):
                    self.export_document(doc)
            
            # Show content preview if available
            if 'content' in doc and doc['content']:
                st.write("**Content Preview:**")
                content = str(doc['content'])
                if len(content) > 1000:
                    st.text_area("", content[:1000] + "...", height=200, disabled=True)
                    st.caption(f"Showing first 1000 characters of {len(content)} total")
                else:
                    st.text_area("", content, height=200, disabled=True)
    
    def delete_document(self, doc_id: str):
        """Delete a specific document"""
        
        try:
            with st.spinner("Deleting document..."):
                result = self.api_client.delete_document(doc_id)
            
            if result["success"]:
                st.success("✅ Document deleted successfully")
                self.refresh_document_list()
            else:
                st.error(f"❌ Failed to delete document: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"Delete error: {str(e)}")
    
    def delete_all_documents(self):
        """Delete all documents"""
        
        try:
            with st.spinner("Deleting all documents..."):
                for doc in st.session_state.uploaded_documents:
                    doc_id = doc.get('id')
                    if doc_id:
                        self.api_client.delete_document(doc_id)
                
                st.success("✅ All documents deleted successfully")
                self.refresh_document_list()
        
        except Exception as e:
            st.error(f"Delete all error: {str(e)}")
    
    def check_processing_status(self, doc_id: str):
        """Check processing status for a document"""
        
        try:
            result = self.api_client.get_document_status(doc_id)
            
            if result["success"]:
                status_data = result["data"]
                st.json(status_data)
            else:
                st.error(f"Failed to get status: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"Status check error: {str(e)}")
    
    def reprocess_document(self, doc_id: str):
        """Reprocess a document"""
        
        st.info("Document reprocessing would be implemented here")
        # This would depend on LightRAG API support for reprocessing
    
    def export_document(self, doc: Dict[str, Any]):
        """Export document data"""
        
        try:
            # Create export data
            export_data = {
                "document_info": doc,
                "export_timestamp": datetime.now().isoformat()
            }
            
            # Convert to JSON
            json_data = json.dumps(export_data, indent=2, default=str)
            
            # Provide download
            st.download_button(
                label="📥 Download Document Data",
                data=json_data,
                file_name=f"document_{doc.get('name', 'unknown')}.json",
                mime="application/json"
            )
        
        except Exception as e:
            st.error(f"Export error: {str(e)}") 