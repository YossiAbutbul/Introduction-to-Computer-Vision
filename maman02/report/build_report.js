// build_report.js - builds report.docx for Maman 12.
// Numbers and figures are read from ../outputs (copied from Google Drive),
// so they always match the last Colab run.
// Run from maman02/report:  node build_report.js  [other_name.docx]

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, WidthType, BorderStyle, HeadingLevel, PageBreak, Footer, PageNumber, LevelFormat,
} = require("docx");

const OUT = path.join(__dirname, "..", "outputs");
const FONT = "Calibri";
const SIZE = 22;                   // 11 pt

// ---------------------------------------------------------------- data

const readJson = (...p) => JSON.parse(fs.readFileSync(path.join(OUT, ...p), "utf8"));
const f = (x, d) => Number(x).toFixed(d);
const pct = (x, d = 1) => f(100 * x, d) + "%";
const int = (x) => Number(x).toLocaleString("en-US");

// split sizes from task0_data.py
const split = readJson("data", "trainval_split.json");
const summary = fs.readFileSync(path.join(OUT, "data", "split_summary.txt"), "utf8").split(/\r?\n/);
const nTest = Number(summary[0].match(/test (\d+)/)[1]);
const dogFrac = (name) => Number(summary.find((l) => l.startsWith(name + ":")).match(/dog fraction ([\d.]+)/)[1]);

const t1 = readJson("task1_cnn", "metrics.json");
const t1h = readJson("task1_cnn", "history.json");

// ---------------------------------------------------------------- text helpers

// Word reverses "224×224" if the whole run is right-to-left, so every string
// is split into Hebrew (RTL) parts and Latin / number (LTR) parts.
const HEB = /[֐-׿]/;
const LTR_CHAR = /[A-Za-z0-9²³×%°_]/;

function splitDirections(text) {
  const parts = [];
  let i = 0;
  while (i < text.length) {
    if (LTR_CHAR.test(text[i])) {
      let j = i, last = i;
      while (j < text.length && !HEB.test(text[j])) {
        // ". " ": " ", " ends a phrase - don't let a Latin block cross it
        if (/[.,:;]/.test(text[j]) && text[j + 1] === " ") break;
        if (text[j] === " " && text[j + 1] === "(") break;
        if (LTR_CHAR.test(text[j])) last = j;
        j++;
      }
      parts.push({ s: text.slice(i, last + 1), rtl: false });
      i = last + 1;
    } else {
      let j = i;
      while (j < text.length && !LTR_CHAR.test(text[j])) j++;
      parts.push({ s: text.slice(i, j), rtl: true });
      i = j;
    }
  }
  return parts;
}

const textRuns = (text, opts = {}) => splitDirections(text).map((p) =>
  new TextRun({ text: p.s, rightToLeft: p.rtl, font: FONT, size: SIZE, sizeComplexScript: SIZE, ...opts }));

const flat = (content, opts) => (Array.isArray(content) ? content : [content])
  .flatMap((c) => (typeof c === "string" ? textRuns(c, opts) : [c]));

function para(content, opts = {}) {
  return new Paragraph({
    bidirectional: true,
    spacing: { after: 100, line: 276 },
    ...opts,
    children: flat(content),
  });
}

function li(content) {
  return new Paragraph({
    bidirectional: true,
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60, line: 276 },
    children: flat(content),
  });
}

function label(text) {
  return para(textRuns(text, { bold: true, boldComplexScript: true }), { spacing: { before: 160, after: 60 }, keepNext: true });
}

const todo = (text) => textRuns(text, { highlight: "yellow" });

function heading(text, level) {
  return new Paragraph({ heading: level, bidirectional: true, children: textRuns(text) });
}

function caption(content) {
  return para(flat(content, { size: 18, sizeComplexScript: 18, italics: true }),
    { alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200 } });
}

// ---------------------------------------------------------------- tables and images

const line = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const borders = { top: line, bottom: line, left: line, right: line };

function cell(content, width, isHead) {
  const opts = { size: 19, sizeComplexScript: 19, bold: isHead, boldComplexScript: isHead };
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    margins: { top: 30, bottom: 30, left: 80, right: 80 },
    children: [new Paragraph({
      bidirectional: true,
      alignment: AlignmentType.CENTER,
      children: flat(typeof content === "number" ? String(content) : content, opts),
    })],
  });
}

// widths are relative, the table is centred
function table(header, rows, widths) {
  const w = widths.map((x) => Math.round(x * 1000));
  const total = w.reduce((a, b) => a + b, 0);
  return new Table({
    visuallyRightToLeft: true,
    alignment: AlignmentType.CENTER,
    width: { size: total, type: WidthType.DXA },
    columnWidths: w,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, w[i], true)) }),
      ...rows.map((r) => new TableRow({ children: r.map((c, i) => cell(c, w[i], false)) })),
    ],
  });
}

// width and height of a PNG from its header
function pngSize(file) {
  const b = fs.readFileSync(file);
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}

// one or more images side by side, each widthPx wide (height keeps the aspect)
function images(files, widthPx) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120 },
    keepNext: true,
    children: files.map((rel) => {
      const file = path.join(OUT, rel);
      const [w, h] = pngSize(file);
      return new ImageRun({ type: "png", data: fs.readFileSync(file),
        transformation: { width: widthPx, height: Math.round(widthPx * h / w) } });
    }),
  });
}

const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

// ---------------------------------------------------------------- content

const body = [];
const add = (...x) => body.push(...x);
let fig = 0;
const figNo = () => ++fig;

// ----- title page
add(
  para(textRuns("מבוא לראייה ממוחשבת (22928)", { size: 36, sizeComplexScript: 36, bold: true, boldComplexScript: true }),
    { alignment: AlignmentType.CENTER, spacing: { before: 3600, after: 200 } }),
  para(textRuns("ממ\"ן 12, סמסטר 2027א", { size: 28, sizeComplexScript: 28 }),
    { alignment: AlignmentType.CENTER, spacing: { after: 1600 } }),
  para(["שם: ", ...todo("[שם מלא]")], { alignment: AlignmentType.CENTER }),
  para(["ת.ז.: ", ...todo("[מספר]")], { alignment: AlignmentType.CENTER }),
  pageBreak(),
);

// ----- code and data
const nTrain = split.train.length, nVal = split.val.length;
add(
  heading("הקוד והנתונים", HeadingLevel.HEADING_1),
  li("הרצתי הכל ב-Google Colab על GPU מסוג " + t1.gpu + " (המחברת colab_run.ipynb). התוצאות נשמרו ב-Google Drive."),
  li("הפונקציות המשותפות נמצאות ב-utils: טעינת הנתונים (data.py), המודל (models.py), לולאת האימון (train.py), המדדים והגרפים."),
  table(["קובץ", "משימה"], [
    ["task0_data.py", "הכנת הנתונים"],
    ["task1_cnn.py", "1.1"],
  ], [3.2, 1.6]),
  para(""),

  heading("חלוקת הנתונים", HeadingLevel.HEADING_2),
  li("השתמשתי בחלוקה הרשמית: " + int(nTrain + nVal) + " תמונות trainval ו-" + int(nTest) + " תמונות test."),
  li("את trainval חילקתי ל-" + int(nTrain) + " אימון ו-" + int(nVal) + " ולידציה (80/20). בכל אחד מ-37 הגזעים ערבבתי את התמונות (seed=0) והעברתי 20% לולידציה."),
  li("החלוקה נשמרת בקובץ splits/trainval_split.json וכל המשימות טוענות אותו, כך שכולן עובדות על אותן תמונות."),
  li("את התוויות קראתי מקבצי annotations/trainval.txt ו-test.txt. מספר הגזע הוא 0 עד 36, ומין החיה 0 לחתול ו-1 לכלב (כלב = מחלקה חיובית)."),
  li("הנתונים לא מאוזנים: " + pct(dogFrac("train")) + " מתמונות האימון הן כלבים (" + pct(dogFrac("test")) + " ב-test). מסווג שתמיד עונה \"כלב\" יקבל " + pct(dogFrac("test")) + " accuracy, ולכן חשוב להסתכל גם על המדדים האחרים."),
  images(["data/breed_counts.png"], 620),
  caption("איור " + figNo() + ": מספר התמונות בכל גזע באימון ובולידציה."),

  heading("עיבוד התמונות", HeadingLevel.HEADING_2),
  li("כל התמונות מנורמלות לפי הממוצע וסטיית התקן של ImageNet."),
  li("באימון (augmentation): חיתוך אקראי של 50%-100% מהשטח ושינוי גודל ל-224×224, היפוך אופקי, ושינוי אקראי קטן של הבהירות, הניגודיות והרוויה (עד 0.2)."),
  li("בולידציה וב-test: שינוי גודל הצלע הקצרה ל-255 וחיתוך מרכזי של 224×224."),
  images(["data/samples.png"], 560),
  caption("איור " + figNo() + ": שורה עליונה - התמונה המקורית, שתי השורות התחתונות - שתי הגרלות של ה-augmentation."),
  pageBreak(),
);

// ---------------------------------------------------------------- task 1.1

const cm = t1.confusion_matrix;               // rows true cat/dog, cols predicted
const nCat = cm[0][0] + cm[0][1], nDog = cm[1][0] + cm[1][1];
const best = t1h.best_epoch;

add(
  heading("משימה 1.1 - רשת קונבולוציה מאפס, חתול או כלב", HeadingLevel.HEADING_1),

  heading("הארכיטקטורה", HeadingLevel.HEADING_2),
  li("בניתי רשת בסגנון ResNet קטן: stem שמקטין את התמונה פי 4, ארבעה שלבים של residual blocks, וראש סיווג."),
  li("כל residual block הוא שתי קונבולוציות 3×3 עם BatchNorm ו-ReLU וחיבור מעקף: relu(x + F(x)). בבלוק הראשון של שלבים 2-4 הקונבולוציה עם stride 2, ובמעקף יש קונבולוציה 1×1 עם stride 2 כדי שהגדלים יתאימו."),
  li("בסוף global average pooling במקום flatten, כך שלראש יש רק 257 פרמטרים."),
  table(["שלב", "שכבות", "גודל הפלט"], [
    ["קלט", "", "224×224×3"],
    ["stem", "conv 3×3 stride 2, conv 3×3, max pool stride 2", "56×56×32"],
    ["שלב 1", "2 residual blocks", "56×56×32"],
    ["שלב 2", "2 residual blocks (הראשון stride 2)", "28×28×64"],
    ["שלב 3", "2 residual blocks (הראשון stride 2)", "14×14×128"],
    ["שלב 4", "2 residual blocks (הראשון stride 2)", "7×7×256"],
    ["ראש", "global avg pool, dropout 0.3, linear", "1 (logit)"],
  ], [1.2, 4.2, 1.6]),
  para(""),
  li("הפלט הוא logit אחד, ואחרי sigmoid מתקבלת ההסתברות שבתמונה יש כלב."),
  li("בחרתי רשת קטנה (כ-2.8 מיליון פרמטרים) כי יש רק " + int(nTrain) + " תמונות אימון, ורשת גדולה יותר תיטה יותר ל-overfitting."),

  heading("האימון", HeadingLevel.HEADING_2),
  table(["פרמטר", "ערך"], [
    ["loss", "BCEWithLogitsLoss"],
    ["optimizer", "AdamW, lr=0.001, weight decay=0.05"],
    ["learning rate schedule", "cosine, מ-0.001 עד 0"],
    ["epochs", String(t1.epochs)],
    ["batch size", "64"],
    ["רגולריזציה", "augmentation, dropout 0.3, weight decay"],
    ["דיוק חישוב", "float16 (mixed precision) על ה-GPU"],
  ], [2.2, 3.6]),
  para(""),
  li("אחרי כל epoch חישבתי loss ו-accuracy על הולידציה ושמרתי את המשקלות של ה-epoch עם ה-loss הנמוך ביותר בולידציה. ה-test שימש רק להערכה הסופית, פעם אחת."),
  li("ה-epoch שנבחר: " + best + " מתוך " + t1.epochs + "."),
  images(["task1_cnn/curves.png"], 620),
  caption("איור " + figNo() + ": loss ו-accuracy באימון ובולידציה. הקו המקווקו הוא ה-epoch שנבחר."),
  li("האימון והולידציה משתפרים יחד, ובסוף הפער ביניהם קטן (" + pct(t1h.train_acc[best - 1]) + " מול " + pct(t1h.val_acc[best - 1]) + " accuracy), כלומר אין overfitting משמעותי."),
  li("בין epoch 10 ל-35 יש בולידציה קפיצות חדות (למשל ה-loss ב-epoch 27). אני חושב שזה בגלל שה-learning rate עדיין גבוה, והסטטיסטיקות של BatchNorm שנשמרו באימון לא מתאימות למשקלות שהשתנו הרבה. לא בדקתי את זה. כשה-learning rate קטן (אחרי epoch 35) העקומה מתייצבת."),
  pageBreak(),

  heading("התוצאות על ה-test", HeadingLevel.HEADING_2),
  table(["מדד", "ערך"], [
    ["Accuracy", pct(t1.accuracy)],
    ["Precision", pct(t1.precision)],
    ["Recall", pct(t1.recall)],
    ["F1", f(t1.f1, 3)],
    ["ROC-AUC", f(t1.roc_auc, 3)],
    ["Average Precision", f(t1.average_precision, 3)],
  ], [2.2, 1.6]),
  para(""),
  images(["task1_cnn/confusion.png"], 300),
  caption("איור " + figNo() + ": מטריצת בלבול על ה-test (סף 0.5)."),
  li("הרשת טועה יותר על חתולים: " + cm[0][1] + " מתוך " + int(nCat) + " חתולים (" + pct(cm[0][1] / nCat) + ") סווגו ככלב, ורק " + cm[1][0] + " מתוך " + int(nDog) + " כלבים (" + pct(cm[1][0] / nDog) + ") סווגו כחתול."),
  li("כנראה בגלל שיש פי 2 כלבים באימון, וכשהרשת לא בטוחה היא נוטה לכלב. זה גם מסביר למה ה-Recall גבוה מה-Precision."),
  images(["task1_cnn/roc.png", "task1_cnn/pr.png"], 300),
  caption("איור " + figNo() + ": עקומת ROC (משמאל) ועקומת Precision-Recall (מימין) על ה-test."),

  heading("גודל, זמן וזיכרון", HeadingLevel.HEADING_2),
  table(["", "ערך"], [
    ["מספר פרמטרים", int(t1.params)],
    ["FLOPs לתמונה (224×224)", f(t1.flops / 1e9, 2) + " G"],
    ["זמן אימון (" + t1.epochs + " epochs, כולל ולידציה)", f(t1.train_time_s / 60, 1) + " דקות"],
    ["זיכרון GPU מקסימלי", f(t1.peak_gpu_mem_mb, 0) + " MB"],
  ], [3.4, 1.8]),
  para(""),
  li("את ה-FLOPs ספרתי עם torch.utils.flop_counter על תמונה אחת (כפל-חיבור = 2 FLOPs)."),
  li("הזיכרון הוא torch.cuda.max_memory_allocated, כלומר מה ש-PyTorch הקצה בפועל. זה פחות מ-4% מה-15GB של ה-T4."),
  li("epoch לקח בערך 25 שניות. אני חושב שרוב הזמן הלך על קריאת התמונות ב-2 ליבות ה-CPU של Colab ולא על ה-GPU, אבל לא מדדתי את זה."),
);

// ---------------------------------------------------------------- document

const headingStyle = (id, name, size, before, outline) => ({
  id, name, basedOn: "Normal", next: "Normal", quickFormat: true,
  run: { font: FONT, size, sizeComplexScript: size, bold: true, boldComplexScript: true, color: "000000" },
  paragraph: { spacing: { before, after: 120 }, keepNext: true, outlineLevel: outline },
});

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: SIZE, sizeComplexScript: SIZE } } },
    paragraphStyles: [
      headingStyle("Heading1", "Heading 1", 32, 240, 0),
      headingStyle("Heading2", "Heading 2", 26, 200, 1),
      headingStyle("Heading3", "Heading 3", 24, 160, 2),
    ],
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 360, hanging: 260 } } } }],
    }],
  },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } },
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }),
    },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, process.argv[2] || "report.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes");
});
