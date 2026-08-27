import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const payloadPath = process.argv[2];
if (!payloadPath) throw new Error("Usage: node Q2_workbook.mjs <payload.json>");
const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
const workbook = Workbook.create();
workbook.comments.setSelf({ displayName: "User" });

const COLORS = {
  header: "#0F766E",
  headerText: "#FFFFFF",
  pale: "#ECFDF5",
  light: "#F8FAFC",
  accent: "#DCFCE7",
  warn: "#FEF3C7",
  border: "#CBD5E1",
  text: "#0F172A",
};

function colName(index) {
  let n = index + 1;
  let out = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    out = String.fromCharCode(65 + rem) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}

function sanitizeTableName(name, index) {
  return `Q2Table_${index}_${Array.from(name).map((ch) => /[A-Za-z0-9]/.test(ch) ? ch : "").join("") || "Data"}`;
}

function widthForColumn(rows, col) {
  let maxLen = 6;
  for (const row of rows.slice(0, 300)) {
    const value = row[col];
    const len = value == null ? 0 : String(value).length;
    maxLen = Math.max(maxLen, Math.min(len, 50));
  }
  return Math.max(10, Math.min(42, maxLen * 1.15 + 2));
}

let sheetIndex = 0;
for (const [sheetName, rows] of Object.entries(payload.sheets)) {
  const sheet = workbook.worksheets.add(sheetName);
  sheet.showGridLines = false;
  if (!rows.length || !rows[0].length) continue;
  const rowCount = rows.length;
  const colCount = Math.max(...rows.map((r) => r.length));
  const padded = rows.map((r) => Array.from({ length: colCount }, (_, i) => r[i] ?? null));
  const end = `${colName(colCount - 1)}${rowCount}`;
  sheet.getRange(`A1:${end}`).values = padded;
  sheet.getRange(`A1:${end}`).format = {
    font: { name: "Microsoft YaHei", size: 10, color: COLORS.text },
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${colName(colCount - 1)}1`).format = {
    fill: COLORS.header,
    font: { name: "Microsoft YaHei", size: 10, bold: true, color: COLORS.headerText },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "medium", color: COLORS.header },
    rowHeight: 28,
  };
  if (rowCount > 1) {
    sheet.getRange(`A2:${end}`).format.borders = {
      insideHorizontal: { style: "thin", color: "#E2E8F0" },
      bottom: { style: "thin", color: COLORS.border },
    };
    sheet.getRange(`A2:${end}`).format.wrapText = true;
  }
  for (let col = 0; col < colCount; col++) {
    sheet.getRange(`${colName(col)}1:${colName(col)}${rowCount}`).format.columnWidth = widthForColumn(padded, col);
  }
  sheet.freezePanes.freezeRows(1);
  if (rowCount > 1 && colCount > 1) {
    const table = sheet.tables.add(`A1:${end}`, true, sanitizeTableName(sheetName, sheetIndex));
    table.style = "TableStyleMedium2";
    table.showBandedRows = true;
    table.showFilterButton = true;
  }
  sheetIndex += 1;
}

// 关键答案保持公式可追溯。
const summary = workbook.worksheets.getItem("题目答案汇总");
summary.getRange("B3").formulas = [["='单角度厚度'!C4"]];
summary.getRange("B4").formulas = [["='单角度厚度'!E4"]];
summary.getRange("B5").formulas = [["='单角度厚度'!F4"]];
summary.getRange("B7").formulas = [["='单角度厚度'!C7"]];
summary.getRange("B8").formulas = [["='单角度厚度'!E7"]];
summary.getRange("B9").formulas = [["='单角度厚度'!F7"]];
summary.getRange("B10").formulas = [["='双角度联合结果'!B2"]];

const bootRows = payload.sheets["Bootstrap结果"];
const jointBoot = [];
for (let i = 1; i < bootRows.length; i++) if (bootRows[i][0] === "双角度联合") jointBoot.push(i + 1);
if (jointBoot.length) {
  summary.getRange("B11").formulas = [[`=PERCENTILE.INC('Bootstrap结果'!$C$${jointBoot[0]}:$C$${jointBoot.at(-1)},0.025)`]];
  summary.getRange("B12").formulas = [[`=PERCENTILE.INC('Bootstrap结果'!$C$${jointBoot[0]}:$C$${jointBoot.at(-1)},0.975)`]];
}
const sensHeader = payload.sheets["敏感性分析"][0];
const sensJointCol = sensHeader.indexOf("联合厚度_um");
if (sensJointCol >= 0) {
  const c = colName(sensJointCol);
  const last = payload.sheets["敏感性分析"].length;
  summary.getRange("B13").formulas = [[`=MIN('敏感性分析'!$${c}$2:$${c}$${last},B11)`]];
  summary.getRange("B14").formulas = [[`=MAX('敏感性分析'!$${c}$2:$${c}$${last},B12)`]];
}
summary.getRange("B15").formulas = [["=ABS(B3-B7)/AVERAGE(B3,B7)"]];
summary.getRange("B16").formulas = [["=B10"]];
summary.getRange("B2:B16").format.numberFormat = "0.000000";
summary.getRange("B15").format.numberFormat = "0.0000%";
summary.getRange("A10:C16").format.fill = COLORS.pale;
summary.getRange("A16:C16").format = {
  fill: COLORS.accent,
  font: { name: "Microsoft YaHei", size: 11, bold: true, color: "#14532D" },
  borders: { preset: "outside", style: "medium", color: "#16A34A" },
};
summary.getRange("A17:C17").format.fill = COLORS.light;
summary.getRange("A18:C18").format.fill = COLORS.warn;
summary.getRange("A1:C18").format.rowHeight = 25;
summary.getRange("A17:C17").format.rowHeight = 62;
summary.getRange("A18:C18").format.rowHeight = 92;
summary.getRange("A1:A18").format.columnWidth = 29;
summary.getRange("B1:B18").format.columnWidth = 24;
summary.getRange("C1:C18").format.columnWidth = 48;

// 常见数值列的显示精度。
for (const sheetName of ["峰值位置", "谷值位置", "单角度厚度", "双角度联合结果", "Bootstrap结果", "敏感性分析", "诊断指标"]) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const used = sheet.getUsedRange();
  used.format.numberFormat = "0.000000";
  sheet.getRange(`A1:${colName(payload.sheets[sheetName][0].length - 1)}1`).format.numberFormat = "@";
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(payload.output);

const summaryInspect = await workbook.inspect({
  kind: "table",
  sheetId: "题目答案汇总",
  range: "A1:C18",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 5,
  maxChars: 5000,
});
console.log(summaryInspect.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

if (process.env.Q2_VERIFY === "1") {
  const previewDir = path.join(path.dirname(payload.output), "_workbook_previews");
  await fs.mkdir(previewDir, { recursive: true });
  for (const sheetName of Object.keys(payload.sheets)) {
    const rows = payload.sheets[sheetName];
    const colCount = rows[0]?.length ?? 1;
    const renderRows = Math.min(rows.length, 40);
    const preview = await workbook.render({
      sheetName,
      range: `A1:${colName(colCount - 1)}${renderRows}`,
      scale: 1,
      format: "png",
    });
    const safe = sheetName.replace(/[\\/:*?"<>|]/g, "_");
    await fs.writeFile(path.join(previewDir, `${safe}.png`), new Uint8Array(await preview.arrayBuffer()));
  }
}
await fs.rm(`${payload.output}.inspect.ndjson`, { force: true });
