questions_data = [
    {
        "name": "Summary Statistics",
        "description": """
                <p>
                    In the field of structural biology, understanding the three-dimensional structure of membrane proteins is crucial for unraveling their functions and developing targeted therapies. Methods such as X-ray crystallography, electron microscopy (EM), and nuclear magnetic resonance (NMR) are used to determine the structures of these proteins. A cumulative bar chart has been created to show the effectiveness of each method in resolving membrane proteins.
                </p>
            """,
        "image": "cell_membrane_image.jpg",
        "questions": [
            {
                "text": """
                    Which method appears to be most used ?
                """,
                "type": "training",
                "item_order": 1,
                "hints": """
                    <p>
                        <b>Hint(s)</b>
                        <ol class='ml-5'>
                            <li> Please hover over the chart with your mouse to see more information. </li>
                        </ol>
                    </p>
                """,
                "instruction": """
                    <p>
                        <b>Instruction</b>
                        <ol class='ml-5'>
                            <li> Click the <strong>“Group by”</strong> field and select the “Experimental Method” option. </li>
                        </ol>
                    </p>
                """,
                "options": [
                    {"text": "   A. X-ray Crystallography", "is_correct": True},
                    {"text": "   B. Nuclear Magnetic Resonance (NMR)", "is_correct": False},
                    {"text": "   C. Cryo-Electron Microscopy (Cryo-EM)", "is_correct": False},
                    {"text": "   D. Multi-methods", "is_correct": False},
                ],
                "filter_tool": [
                    {
                        "title": "Group By: ",
                        "name": "groupby",
                        "selectedOption": "", 
                        "options": [
                            {"text": "Select option", "value": ""},
                            {"text": "Experimental Method", "value": "rcsentinfo_experimental_method"},
                            {"text": "Membrane Protein Structure Group", "value": "group"},
                            {"text": "Taxonomic Domain", "value": "taxonomic_domain"}
                        ]
                    }
                ],
            },
            {
                "text": """
                    Which experimental method appears to be growing faster now ?
                """,
                "type": "test",
                "item_order": 2,
                "hints": """""",
                "options": [
                    {"text": "  A. X-ray Crystallography", "is_correct": False},
                    {"text": "  B. Nuclear Magnetic Resonance (NMR)", "is_correct": False},
                    {"text": "  C. Cryo-Electron Microsopy (Cryo-EM)", "is_correct": True},
                    {"text": "  D. Multi-Methods", "is_correct": False},
                ],
                "filter_tool": [
                    {
                        "title": "Group By: ", 
                        "name": "groupby",
                        "selectedOption": "", 
                        "options": [
                            {"text": "Select option", "value": ""},
                            {"text": "Experimental Method", "value": "rcsentinfo_experimental_method"},
                            {"text": "Membrane Protein Structure Group", "value": "group"},
                            {"text": "Taxonomic Domain", "value": "taxonomic_domain"}
                        ]
                    }
                ],
            },
        ]
    },
    {
        "name": "Outliers Identification/Detection",
        "description": """
                <p class="show-for-question-3-4-and-5">
                    In structural biology, detailed molecular information is crucial. 
                    Techniques like X-ray crystallography, electron microscopy, and NMR 
                    spectroscopy capture structures at various resolutions, ensuring reliable data. 
                    Protein structure databases aid in understanding functions and drug development, 
                    though annotation differences can cause discrepancies. 
                    Outlier detection methods, such as scatter plot matrix (SPLOM), box plots, and machine learning, 
                    identify deviations, improving precision.
                </p>
                
                <p class="show-for-question-5-and-6" style="display:none">
                    Outlier detection involves identifying unusual data points that may be errors or unique cases requiring 
                    closer examination.
                </p>
                <p class="show-for-question-5-and-6" style="display:none">
                    We have selected DBSCAN (Density-Based Spatial Clustering of Applications with Noise) for this task. 
                    Unlike traditional statistical methods, DBSCAN uses data density to detect outliers. 
                    It groups closely located points into clusters and identifies points in sparse regions as outliers.
                </p>
                <p class="show-for-question-6" style="display:none">
                    A <strong>SPLOM</strong> is a grid of scatterplots that shows the relationships between pairs of variables in a dataset.
                    Each cell contains a scatterplot for two variables, allowing for visual inspection of patterns and 
                    potential outliers. It is a useful tool for exploratory data analysis.
                </p>
            """,
        "image": "cell_membrane_image.jpg",
        "questions": [
            {
                "text": """
                    Identify membrane protein structure groups that contain outliers.
                """,
                "type": "training",
                "item_order": 3,
                "hints": """
                    <p>
                        <b>Hint(s)</b>
                        <ol class='ml-5'>
                            <li> Please hover over the chart with your mouse to see more information.</li>
                        </ol>
                    </p>
                """,
                "instruction": """
                    <p>
                        <b>Instruction (s)</b>
                        <ol class='ml-5'>
                            <li> 
                                Click the <strong>“Experimental Method”</strong> field and select the X-ray crystallography option.
                            </li>
                        </ol>
                    </p>
                """,
                "options": [
                    {"text": "   A. 1 & 2", "is_correct": False},
                    {"text": "   B. 2 & 3", "is_correct": True},
                    {"text": "   C. 1 & 3", "is_correct": False},
                    {"text": "   D. None", "is_correct": False},
                ],
                "filter_tool": [
                    {
                        "title": "Experimental Method: ", 
                        "name": "methods",
                        "selectedOption": "", 
                        "options": [
                            {"text": "Select option", "value": ""},
                            {"text": "X-ray crystallography", "value": "X-ray"},
                            {"text": "Electron microscopy (EM)", "value": "EM"},
                            {"text": "Nuclear magnetic resonance (NMR)", "value": "NMR"}
                        ]
                    }
                ],
            },
            {
                "text": """
                    Study the variations in resolution values using <strong>electron microscopy (EM)</strong>, specifically focusing on the initial group in the boxplot illustrating <strong>Monotopic Membrane Protein Structures</strong>. How many outliers are evident within this context?
                """,
                "type": "test",
                "item_order": 4,
                "hints": """""",
                "options": [
                    {"text": "   A. 1 (3J0J)", "is_correct": False},
                    {"text": "   B. 1 (6ZG5)", "is_correct": True},
                    {"text": "   C. 1 (4UQJ)", "is_correct": False},
                    {"text": "   D. 2 (7RI7, 7LHI)", "is_correct": False},
                ],
                "filter_tool": [
                    {
                        "title": "Experimental Method: ", 
                        "name": "methods",
                        "selectedOption": "", 
                        "options": [
                            {"text": "Select option", "value": ""},
                            {"text": "X-ray crystallography", "value": "X-ray"},
                            {"text": "Electron microscopy (EM)", "value": "EM"},
                            {"text": "Nuclear magnetic resonance (NMR)", "value": "NMR"}
                        ]
                    }
                ],
            },
            {
                "text": """
                    How many outliers in the scatter plot matrix (SPLOM) were not identified by MetaMP using the DBSCAN ?
                """,
                "type": "training",
                "item_order": 6,
                "hints": """
                    <p>
                        <b>Hint(s)</b>
                        <ol class='ml-5'>
                            <li> Click and drag your mouse over the chart area of interest.</li>
                            <li> Release the mouse to highlight and display data points selected in the other chart, allowing for effective comparison and outlier identification.</li>
                            <li> Click outside the selected area to clear the highlight and return to the full chart view.</li>
                        </ol>
                    </p>
                """,
                "instruction": """
                    <p>
                        <b>Instruction (s)</b>
                        <ol class='ml-5'>
                            <li> 
                                We have chosen default features like 
                                <span title="Mass of the molecule in Daltons"><b>'Molecular Weight'</b></span>, 
                                <span title="Total number of particles used in the reconstruction"><b>'Number of Particles in Reconstruction'</b></span>, 
                                and <span title="Level of detail of the protein structure (measured in Ångströms)"><b>'Resolution'</b></span>
                                as shown in the "Feature" field. 
                                These features are relevant to <strong>Cryo-Electron Microscopy (EM)</strong>.
                            </li>
                        </ol>
                    </p>
                """,
                "options": [
                    {"text": "   A. Fewer than 5", "is_correct": False},
                    {"text": "   B. Around 10", "is_correct": False},
                    {"text": "   C. More than 10", "is_correct": False},
                    {"text": "   D. None", "is_correct": True},
                ],
                "filter_tool": [
                    {
                        "title": "Features: ", 
                        "name": "features",
                        "selectedOption": "", 
                        "options": [
                            {"text": "Molecular Weight", "value": "emt_molecular_weight"},
                            {"text": "Number of Particles in Reconstruction", "value": "reconstruction_num_particles"},
                            {"text": "Resolution", "value": "processed_resolution"}
                        ]
                    }
                ],
            },
            {
                "text": """
                    Do you observe any outliers in the scatter plot matrix (SPLOM) that the MetaMP failed to identify using DBSCAN ?
                """,
                "type": "test",
                "item_order": 7,
                "hints": """
                    <p>
                        <b>Hint(s)</b>
                        <ol class='ml-5'>
                            <li> Click and drag your mouse over the chart area of interest.</li>
                            <li> Release the mouse to highlight and display data points selected in the other chart, allowing for effective comparison and outlier identification.</li>
                            <li> Click outside the selected area to clear the highlight and return to the full chart view.</li>
                        </ol>
                    </p>
                """,
                "instruction": """
                    <p>
                        <b>Instruction (s)</b>
                        <ol class='ml-5'>
                            <li> 
                                We have selected default features such as 
                                <span title="The Matthews coefficient, in crystallography, is a measure used to estimate the packing density of proteins in a crystal lattice"><b>'Crystal Density Matthews'</b></span>,
                                <span title="Mass of the molecule in Daltons"><b>'Molecular Weight'</b></span>, 
                                and <span title="Level of detail of the protein structure (measured in Ångströms)"><b>'Resolution'</b></span>
                                as seen in the "Feature" field. 
                                These features are relevant to X-ray Crystallography (X-ray).
                            </li>
                        </ol>
                    </p>
                """,
                "options": [
                    {"text": "   A. Yes", "is_correct": False},
                    {"text": "   B. No", "is_correct": True},
                    {"text": "   C. Not sure", "is_correct": False},
                ],
                "filter_tool": [
                    {
                        "title": "Features: ", 
                        "name": "features",
                        "selectedOption": "", 
                        "options": [
                            {"text": "Crystal Density Matthews", "value": "crystal_density_matthews"},
                            {"text": "Molecular Weight", "value": "molecular_weight"},
                            {"text": "Resolution", "value": "processed_resolution"}
                        ]
                    }
                ],
            }
        ]
    }
]
    