const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const schemaDir = './processed_schemas'; // Update with your schema directory
const outputDir = './converted_processed_schemas'; // Update with the output directory

//https://github.com/sourcemeta-research/alterschema

if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir);
}

// Function to determine the draft version from the $schema field
const getDraftVersion = (schema) => {
  const schemaUrl = schema.$schema;
  if (schemaUrl) {
    if (schemaUrl.includes('2020-12')) {
      return '2020-12';
    } else if (schemaUrl.includes('2019-09')) {
      return '2019-09';
    } else if (schemaUrl.includes('draft-07')) {
      return 'draft7';
    } else if (schemaUrl.includes('draft-06')) {
      return 'draft6';
    } else if (schemaUrl.includes('draft-04')) {
      return 'draft4';
    } else if (schemaUrl.includes('draft-03')) {
      return 'draft3';
    }
  }
  return 'draft7';
};

// Convert schema or copy if already in 2020-12
const processSchema = (filePath) => {
  const schema = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  const currentDraft = getDraftVersion(schema);
  const outputFilePath = path.join(outputDir, path.basename(filePath));

  if (currentDraft === '2020-12') {
    console.log(`Schema ${filePath} is already in draft-2020-12. Copying to output directory.`);
    fs.copyFileSync(filePath, outputFilePath);
    console.log(`Copied: ${filePath} → ${outputFilePath}`);
    return;
  }

  if (currentDraft) {
    console.log(`Converting ${filePath} from ${currentDraft} to draft-2020-12...`);
    try {
      execSync(`alterschema --from ${currentDraft} --to 2020-12 ${filePath} > ${outputFilePath}`, { stdio: 'inherit' });
      console.log(`Converted: ${filePath} → Saved to: ${outputFilePath}`);
    } catch (error) {
      console.error(`Error converting ${filePath}:`, error.message);
    }
  } else {
    console.log(`Skipping ${filePath}, no recognized draft version found.`);
  }
};

// Process all schema files in the directory
fs.readdirSync(schemaDir).forEach((fileName) => {
  if (fileName.endsWith('.json')) {
    const filePath = path.join(schemaDir, fileName);
    processSchema(filePath);
  }
});
