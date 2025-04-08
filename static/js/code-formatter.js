/**
 * Formats and highlights code in the UI
 */
document.addEventListener('DOMContentLoaded', function() {
    // Function to properly format code blocks with syntax highlighting
    window.formatCodeBlock = function(codeElement, codeContent) {
        // Check if the code already has markdown code blocks
        let language = 'python'; // Default language
        let codeText = codeContent;
        
        // Try to extract language and code from markdown code blocks ```language code ```
        const codeBlockRegex = /```(\w+)?\s*([\s\S]*?)```/;
        const match = codeContent.match(codeBlockRegex);
        
        if (match) {
            if (match[1]) {
                language = match[1].toLowerCase();
            }
            codeText = match[2].trim();
        }
        
        // Detect language if not specified in markdown
        if (language === 'python' || language === '') {
            if (codeText.includes('function') && (codeText.includes('{') || codeText.includes('=>'))) {
                language = 'javascript';
            } else if (codeText.includes('public class') || (codeText.includes('{') && codeText.includes(';'))) {
                language = 'java';
            } else if (codeText.includes('#include') && codeText.includes('int main(')) {
                language = 'cpp';
            }
        }
        
        // Create properly formatted code block with syntax highlighting
        const preElement = document.createElement('pre');
        preElement.className = 'language-' + language;
        
        const codeElement2 = document.createElement('code');
        codeElement2.className = 'language-' + language;
        codeElement2.textContent = codeText;
        
        preElement.appendChild(codeElement2);
        
        // Clear the container and add the new elements
        codeElement.innerHTML = '';
        codeElement.appendChild(preElement);
        
        // Apply syntax highlighting
        Prism.highlightElement(codeElement2);
    };
    
    // Override code generation handler
    const originalCodeHandler = function() {
        // Store the original handler
        const originalHandlers = {};
        
        // Override output rendering for code blocks
        const codeForm = document.getElementById('code-form');
        if (codeForm) {
            const originalHandler = codeForm.onsubmit;
            originalHandlers.code = originalHandler;
            
            codeForm.onsubmit = function(e) {
                e.preventDefault();
                
                const prompt = document.getElementById('code-prompt').value;
                const maxLength = document.getElementById('code-max-length').value;
                const temperature = document.getElementById('code-temperature').value;
                
                const generateBtn = document.getElementById('generate-code-btn');
                generateBtn.disabled = true;
                generateBtn.classList.add('loading-btn');
                const originalText = generateBtn.textContent;
                generateBtn.setAttribute('data-original-text', originalText);
                generateBtn.innerHTML = '<span style="visibility: hidden;">' + originalText + '</span>';
                
                fetch('/api/generate_code', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        max_length: maxLength,
                        temperature: temperature
                    }),
                })
                .then(response => response.json())
                .then(data => {
                    generateBtn.disabled = false;
                    generateBtn.classList.remove('loading-btn');
                    generateBtn.textContent = originalText;
                    
                    if (data.status === 'success') {
                        const outputContainer = document.getElementById('code-output');
                        const outputContent = document.getElementById('generated-code');
                        
                        // Use the formatter function
                        formatCodeBlock(outputContent, data.code);
                        outputContainer.classList.remove('d-none');
                        
                        // Add source information
                        const existingSourceInfo = outputContainer.querySelector('.source-info');
                        if (existingSourceInfo) {
                            existingSourceInfo.remove();
                        }
                        
                        const sourceInfo = document.createElement('div');
                        sourceInfo.className = 'source-info mt-3';
                        
                        if (data.source === 'actual_model') {
                            sourceInfo.innerHTML = 
                                '<div class="alert alert-success">' +
                                '<strong><i class="bi bi-check-circle-fill me-2"></i>Using actual Gemma model</strong>' +
                                '<p class="mb-0 small">This response was generated by the actual Gemma model on Hugging Face.</p>' +
                                '</div>';
                        } else {
                            sourceInfo.innerHTML = 
                                '<div class="alert alert-warning">' +
                                '<strong><i class="bi bi-exclamation-triangle-fill me-2"></i>Using simulated response</strong>' +
                                '<p class="mb-0 small">This is a pre-written example. To get actual model responses, provide a ' +
                                'Hugging Face token with access to the Gemma model.</p>' +
                                '</div>';
                        }
                        
                        outputContainer.appendChild(sourceInfo);
                        
                        // Scroll to output
                        outputContainer.scrollIntoView({ behavior: 'smooth' });
                    } else {
                        // Show error
                        const errorElement = document.getElementById('code-error');
                        errorElement.textContent = data.errors ? data.errors.join(', ') : 'An error occurred';
                        errorElement.classList.remove('d-none');
                    }
                })
                .catch(error => {
                    console.error('Error generating code:', error);
                    generateBtn.disabled = false;
                    generateBtn.classList.remove('loading-btn');
                    generateBtn.textContent = originalText;
                    
                    const alertDiv = document.createElement('div');
                    alertDiv.className = 'alert alert-danger alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3';
                    alertDiv.style.zIndex = 1050;
                    alertDiv.innerHTML = `
                        Error generating code. Please try again.
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    `;
                    
                    document.body.appendChild(alertDiv);
                    
                    setTimeout(() => {
                        alertDiv.classList.remove('show');
                        setTimeout(() => {
                            alertDiv.remove();
                        }, 150);
                    }, 3000);
                });
            };
        }
        
        return originalHandlers;
    }();
    
    // If there's existing code content, format it
    const existingCodeBlock = document.getElementById('generated-code');
    if (existingCodeBlock && existingCodeBlock.textContent.trim()) {
        formatCodeBlock(existingCodeBlock, existingCodeBlock.textContent);
    }
    
    // Update the copy function to work with the new code structure
    const copyButtons = document.querySelectorAll('.copy-btn');
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const targetElement = document.getElementById(targetId);
            
            // If it's a code block, get the text from the code element
            let textToCopy = targetElement.textContent;
            if (targetId === 'generated-code' && targetElement.querySelector('code')) {
                textToCopy = targetElement.querySelector('code').textContent;
            }
            
            navigator.clipboard.writeText(textToCopy).then(() => {
                // Change button text temporarily
                const originalText = this.textContent;
                this.textContent = 'Copied!';
                setTimeout(() => {
                    this.textContent = originalText;
                }, 2000);
            }).catch(err => {
                console.error('Error copying text:', err);
            });
        });
    });
});
